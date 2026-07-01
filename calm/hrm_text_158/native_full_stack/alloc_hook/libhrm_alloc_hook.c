#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define HRM_HOOK_MAGIC 0x48524d414c4c484bULL
#define HRM_HOOK_VERSION 1
#define HRM_OWNERSHIP_SLOTS (1u << 22) /* 4M */
#define HRM_FRAME_NET_SLOTS (1u << 16)
#define HRM_RING_SLOTS (1u << 20) /* 1M */
#define HRM_TOP_SITES 32
#define HRM_FRAME_DEPTH 4

typedef struct {
    uintptr_t key;
    uintptr_t key_end;
    uint64_t size;
    uint64_t owner_frame;
    uint32_t flags;
    uint32_t valid;
} hrm_owner_slot_t;

typedef struct {
    uint64_t owner_frame;
    int64_t net_bytes;
    uint64_t gross_alloc;
    uint64_t gross_free;
    uint32_t valid;
} hrm_frame_net_t;

typedef struct {
    uint32_t op;
    uint64_t size;
    uint64_t frames[HRM_FRAME_DEPTH];
} hrm_ring_rec_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t enabled;
    uint32_t prefault_done;
    uint32_t hook_active;
    uint64_t hook_table_start;
    uint64_t hook_table_end;
    uint64_t hook_ring_start;
    uint64_t hook_ring_end;
    uint64_t ring_drop_count;
    uint64_t lock_contention_drop_count;
    uint64_t table_overflow_count;
    uint64_t table_eviction_count;
    uint64_t unknown_free_bytes;
    uint64_t unknown_free_unmeasured_count;
    uint32_t unknown_free_bytes_bounded;
    uint64_t lost_owner_count;
    uint64_t window_net_bytes;
    uint64_t positive_control_hits;
    struct {
        uint64_t owner_frame;
        int64_t net_bytes;
        uint64_t gross_alloc;
        uint64_t gross_free;
        uint64_t count;
    } top[HRM_TOP_SITES];
} hrm_hook_stats_t;

static hrm_owner_slot_t *g_owner_table = NULL;
static hrm_frame_net_t *g_frame_net = NULL;
static hrm_ring_rec_t *g_ring = NULL;
static atomic_uint g_ring_head = 0;
static atomic_bool g_recording_armed = ATOMIC_VAR_INIT(false);
static hrm_hook_stats_t *g_stats = NULL;
static _Thread_local int g_in_hook = 0;
static pthread_mutex_t g_record_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_hook_ready = false;
static bool g_tables_ready = false;
static bool g_env_gate_enabled = false;
static pthread_t g_main_thread;
static bool g_main_thread_set = false;

static void init_hook(void);
static void ensure_tables(void);

static void *(*real_malloc)(size_t) = NULL;
static void *(*real_calloc)(size_t, size_t) = NULL;
static void *(*real_realloc)(void *, size_t) = NULL;
static void (*real_free)(void *) = NULL;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t) = NULL;
static int (*real_munmap)(void *, size_t) = NULL;
static void *(*real_mremap)(void *, size_t, size_t, int, ...) = NULL;
static int (*real_posix_memalign)(void **, size_t, size_t) = NULL;
static void *(*real_aligned_alloc)(size_t, size_t) = NULL;
static void *(*real_memalign)(size_t, size_t) = NULL;
static void *(*real_valloc)(size_t) = NULL;
static void *(*real_pvalloc)(size_t) = NULL;
static size_t (*real_malloc_usable_size)(void *) = NULL;

static bool env_enabled(void) {
    const char *v = getenv("HRM_TEXT_158_PROFILE_ALLOC_HOOK");
    if (!v) return false;
    return v[0] == '1' || (v[0] && (strcmp(v, "true") == 0 || strcmp(v, "on") == 0 || strcmp(v, "yes") == 0));
}

static bool env_rss_enabled(void) {
    const char *v = getenv("HRM_TEXT_158_PROFILE_HOST_RSS");
    if (!v) return false;
    return v[0] == '1' || (v[0] && (strcmp(v, "true") == 0 || strcmp(v, "on") == 0 || strcmp(v, "yes") == 0));
}

static bool recording_armed(void) {
    return g_env_gate_enabled && g_tables_ready && g_stats && g_stats->enabled &&
           atomic_load_explicit(&g_recording_armed, memory_order_acquire);
}

static bool tracking_active(void) {
    return atomic_load_explicit(&g_recording_armed, memory_order_acquire);
}

static bool hook_should_run(void) {
    if (!g_hook_ready) init_hook();
    return recording_armed();
}

static bool record_lock_try(void) {
    return pthread_mutex_trylock(&g_record_lock) == 0;
}

static void record_unlock(void) { pthread_mutex_unlock(&g_record_lock); }

static bool recording_thread_allowed(void) {
    return g_main_thread_set && pthread_equal(pthread_self(), g_main_thread);
}

static bool should_intercept_alloc(void) {
    return tracking_active() && recording_thread_allowed();
}

static bool frame_valid(uintptr_t f) {
    return f > 0x10000;
}

static bool frame_in_skip_module(uintptr_t frame) {
    Dl_info info;
    if (!frame_valid(frame) || dladdr((void *)frame, &info) == 0 || !info.dli_fname) {
        return true;
    }
    const char *name = info.dli_fname;
    return strstr(name, "libhrm_alloc_hook") != NULL || strstr(name, "/libc.") != NULL ||
           strstr(name, "ld-linux") != NULL || strstr(name, "libdl.") != NULL ||
           strstr(name, "libpthread") != NULL || strstr(name, "libgcc") != NULL ||
           strstr(name, "libstdc++") != NULL;
}

static void capture_frames(uintptr_t out[HRM_FRAME_DEPTH]) {
    out[0] = (uintptr_t)__builtin_return_address(1);
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
}

static uint64_t attribute_frame(uintptr_t frames[HRM_FRAME_DEPTH]) {
    for (int i = 0; i < HRM_FRAME_DEPTH; ++i) {
        uintptr_t frame = frames[i];
        if (!frame_valid(frame) || frame_in_skip_module(frame)) continue;
        return (uint64_t)frame;
    }
    for (int i = 0; i < HRM_FRAME_DEPTH; ++i) {
        if (frame_valid(frames[i])) return (uint64_t)frames[i];
    }
    return 0;
}

static uint32_t owner_hash(uintptr_t key) {
    return (uint32_t)((key >> 4) % HRM_OWNERSHIP_SLOTS);
}

static uint32_t frame_hash(uint64_t frame) {
    return (uint32_t)((frame >> 4) % HRM_FRAME_NET_SLOTS);
}

static void frame_net_add(uint64_t owner_frame, int64_t delta, uint64_t alloc_inc, uint64_t free_inc) {
    if (!owner_frame) return;
    uint32_t h = frame_hash(owner_frame);
    for (uint32_t i = 0; i < HRM_FRAME_NET_SLOTS; ++i) {
        uint32_t idx = (h + i) % HRM_FRAME_NET_SLOTS;
        hrm_frame_net_t *slot = &g_frame_net[idx];
        if (!slot->valid || slot->owner_frame == owner_frame) {
            if (!slot->valid) {
                slot->owner_frame = owner_frame;
                slot->valid = 1;
            }
            slot->net_bytes += delta;
            slot->gross_alloc += alloc_inc;
            slot->gross_free += free_inc;
            return;
        }
    }
}

static bool owner_insert(uintptr_t key, uintptr_t key_end, uint64_t size, uint64_t owner_frame) {
    uint32_t h = owner_hash(key);
    for (uint32_t i = 0; i < HRM_OWNERSHIP_SLOTS; ++i) {
        uint32_t idx = (h + i) % HRM_OWNERSHIP_SLOTS;
        hrm_owner_slot_t *slot = &g_owner_table[idx];
        if (!slot->valid || slot->key == key) {
            if (slot->valid && slot->key == key) {
                return true;
            }
            if (!slot->valid) {
                slot->key = key;
                slot->key_end = key_end;
                slot->size = size;
                slot->owner_frame = owner_frame;
                slot->valid = 1;
                return true;
            }
        }
    }
    if (g_stats) {
        g_stats->table_overflow_count++;
        g_stats->lost_owner_count++;
    }
    return false;
}

static hrm_owner_slot_t *owner_lookup(uintptr_t key) {
    uint32_t h = owner_hash(key);
    for (uint32_t i = 0; i < HRM_OWNERSHIP_SLOTS; ++i) {
        uint32_t idx = (h + i) % HRM_OWNERSHIP_SLOTS;
        hrm_owner_slot_t *slot = &g_owner_table[idx];
        if (!slot->valid) return NULL;
        if (slot->key == key) return slot;
    }
    return NULL;
}

static void owner_remove(uintptr_t key) {
    hrm_owner_slot_t *slot = owner_lookup(key);
    if (slot) slot->valid = 0;
}

static void ring_push(uint32_t op, uint64_t size, uintptr_t frames[HRM_FRAME_DEPTH]) {
    uint32_t idx = atomic_fetch_add(&g_ring_head, 1);
    if (idx >= HRM_RING_SLOTS) {
        if (g_stats) g_stats->ring_drop_count++;
        return;
    }
    hrm_ring_rec_t *rec = &g_ring[idx];
    rec->op = op;
    rec->size = size;
    memcpy(rec->frames, frames, sizeof(rec->frames));
}

static void record_alloc(uint32_t op, void *ptr, uint64_t size, uintptr_t frames[HRM_FRAME_DEPTH]) {
    if (!ptr || !size || !hook_should_run()) return;
    if (!record_lock_try()) {
        if (g_stats) g_stats->lock_contention_drop_count++;
        return;
    }
    uint64_t owner = attribute_frame(frames);
    if (owner_insert((uintptr_t)ptr, (uintptr_t)ptr + (uintptr_t)size, size, owner)) {
        frame_net_add(owner, (int64_t)size, size, 0);
        if (g_stats) g_stats->window_net_bytes += size;
        ring_push(op, size, frames);
    }
    record_unlock();
}

static void record_free_ptr(void *ptr) {
    if (!ptr || !hook_should_run()) return;
    if (!record_lock_try()) {
        if (g_stats) g_stats->lock_contention_drop_count++;
        return;
    }
    hrm_owner_slot_t *slot = owner_lookup((uintptr_t)ptr);
    if (slot) {
        frame_net_add(slot->owner_frame, -(int64_t)slot->size, 0, slot->size);
        if (g_stats) g_stats->window_net_bytes -= slot->size;
        owner_remove((uintptr_t)ptr);
        record_unlock();
        return;
    }
    if (g_stats) {
        g_stats->unknown_free_bytes_bounded = 0;
        g_stats->unknown_free_unmeasured_count++;
    }
    record_unlock();
}

static void record_munmap(void *ptr, size_t len) {
    if (!ptr || !hook_should_run()) return;
    if (!record_lock_try()) {
        if (g_stats) g_stats->lock_contention_drop_count++;
        return;
    }
    hrm_owner_slot_t *slot = owner_lookup((uintptr_t)ptr);
    if (slot) {
        frame_net_add(slot->owner_frame, -(int64_t)slot->size, 0, slot->size);
        if (g_stats) g_stats->window_net_bytes -= slot->size;
        owner_remove((uintptr_t)ptr);
        record_unlock();
        return;
    }
    if (g_stats) {
        g_stats->unknown_free_bytes += (uint64_t)len;
        g_stats->unknown_free_bytes_bounded = 1;
    }
    record_unlock();
}

static void resolve_reals(void) {
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_free = dlsym(RTLD_NEXT, "free");
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    real_munmap = dlsym(RTLD_NEXT, "munmap");
    real_mremap = dlsym(RTLD_NEXT, "mremap");
    real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    real_aligned_alloc = dlsym(RTLD_NEXT, "aligned_alloc");
    real_memalign = dlsym(RTLD_NEXT, "memalign");
    real_valloc = dlsym(RTLD_NEXT, "valloc");
    real_pvalloc = dlsym(RTLD_NEXT, "pvalloc");
    real_malloc_usable_size = dlsym(RTLD_NEXT, "malloc_usable_size");
}

static void prefault_region(void *base, size_t len) {
    volatile char *p = (volatile char *)base;
    size_t page = 4096;
    for (size_t off = 0; off < len; off += page) {
        p[off] = 0;
    }
}

static void ensure_tables(void) {
    if (g_tables_ready) return;
    g_env_gate_enabled = env_enabled() && env_rss_enabled();
    if (!g_env_gate_enabled) return;
    if (!real_calloc) resolve_reals();
    size_t owner_bytes = (size_t)HRM_OWNERSHIP_SLOTS * sizeof(hrm_owner_slot_t);
    size_t frame_bytes = (size_t)HRM_FRAME_NET_SLOTS * sizeof(hrm_frame_net_t);
    size_t ring_bytes = (size_t)HRM_RING_SLOTS * sizeof(hrm_ring_rec_t);
    size_t stats_bytes = sizeof(hrm_hook_stats_t);
    int saved_in_hook = g_in_hook;
    g_in_hook = 1;
    g_owner_table = real_calloc(1, owner_bytes);
    g_frame_net = real_calloc(1, frame_bytes);
    g_ring = real_calloc(1, ring_bytes);
    g_stats = real_calloc(1, stats_bytes);
    g_in_hook = saved_in_hook;
    if (!g_owner_table || !g_frame_net || !g_ring || !g_stats) {
        return;
    }
    g_stats->magic = HRM_HOOK_MAGIC;
    g_stats->version = HRM_HOOK_VERSION;
    g_stats->enabled = 1;
    g_stats->hook_active = 0; /* recording disarmed until hrm_alloc_hook_arm() */
    g_stats->unknown_free_bytes_bounded = 1;
    g_stats->hook_table_start = (uint64_t)(uintptr_t)g_owner_table;
    g_stats->hook_table_end = g_stats->hook_table_start + owner_bytes;
    g_stats->hook_ring_start = (uint64_t)(uintptr_t)g_ring;
    g_stats->hook_ring_end = g_stats->hook_ring_start + ring_bytes;
    g_tables_ready = true;
}

static void init_hook(void) {
    if (g_hook_ready) return;
    resolve_reals();
    g_env_gate_enabled = env_enabled() && env_rss_enabled();
    g_hook_ready = true;
}

__attribute__((constructor)) static void hrm_alloc_hook_ctor(void) {
    resolve_reals();
    g_env_gate_enabled = env_enabled() && env_rss_enabled();
    g_hook_ready = true;
}

void *malloc(size_t size) {
    if (!real_malloc) resolve_reals();
    return real_malloc(size);
}

void free(void *ptr) {
    if (!real_free) resolve_reals();
    real_free(ptr);
}

void *calloc(size_t nmemb, size_t size) {
    if (!real_calloc) resolve_reals();
    return real_calloc(nmemb, size);
}

void *realloc(void *ptr, size_t size) {
    if (!real_realloc) resolve_reals();
    return real_realloc(ptr, size);
}

static bool should_intercept_mmap(void) {
    return recording_armed() && recording_thread_allowed();
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (!real_mmap) resolve_reals();
    void *ptr = real_mmap(addr, length, prot, flags, fd, offset);
    if (ptr != MAP_FAILED && fd == -1 && should_intercept_mmap()) {
        uintptr_t frames[HRM_FRAME_DEPTH];
        capture_frames(frames);
        record_alloc(10, ptr, (uint64_t)length, frames);
    }
    return ptr;
}

int munmap(void *addr, size_t length) {
    if (!real_munmap) resolve_reals();
    if (addr && should_intercept_mmap()) {
        record_munmap(addr, length);
    }
    return real_munmap(addr, length);
}

int posix_memalign(void **memptr, size_t alignment, size_t size) {
    if (!real_posix_memalign) resolve_reals();
    return real_posix_memalign(memptr, alignment, size);
}

void *aligned_alloc(size_t alignment, size_t size) {
    if (!real_aligned_alloc) resolve_reals();
    return real_aligned_alloc(alignment, size);
}

void *memalign(size_t alignment, size_t size) {
    if (!real_memalign) resolve_reals();
    return real_memalign(alignment, size);
}

void *valloc(size_t size) {
    if (!real_valloc) resolve_reals();
    return real_valloc(size);
}

void *pvalloc(size_t size) {
    if (!real_pvalloc) resolve_reals();
    return real_pvalloc(size);
}

void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off64_t offset) {
    return mmap(addr, length, prot, flags, fd, (off_t)offset);
}

void *reallocarray(void *ptr, size_t nmemb, size_t size) {
    if (nmemb && size > (~(size_t)0) / nmemb) {
        errno = ENOMEM;
        return NULL;
    }
    return realloc(ptr, nmemb * size);
}

int hrm_alloc_hook_is_active(void) {
    init_hook();
    if (!env_enabled() || !env_rss_enabled()) return 0;
    return g_tables_ready && g_stats && g_stats->enabled ? 1 : 0;
}

int hrm_alloc_hook_is_recording(void) {
    init_hook();
    return recording_armed() ? 1 : 0;
}

void hrm_alloc_hook_arm(void) {
    init_hook();
    if (!g_env_gate_enabled || !g_tables_ready) return;
    g_main_thread = pthread_self();
    g_main_thread_set = true;
    if (g_stats) g_stats->hook_active = 1;
    atomic_store_explicit(&g_recording_armed, true, memory_order_release);
}

void hrm_alloc_hook_disarm(void) {
    atomic_store_explicit(&g_recording_armed, false, memory_order_release);
    if (g_stats) g_stats->hook_active = 0;
}

int hrm_alloc_hook_prefault(void) {
    init_hook();
    if (!env_enabled() || !env_rss_enabled()) return 0;
    ensure_tables();
    if (!g_tables_ready || !g_stats) return 0;
    size_t owner_bytes = (size_t)HRM_OWNERSHIP_SLOTS * sizeof(hrm_owner_slot_t);
    size_t frame_bytes = (size_t)HRM_FRAME_NET_SLOTS * sizeof(hrm_frame_net_t);
    size_t ring_bytes = (size_t)HRM_RING_SLOTS * sizeof(hrm_ring_rec_t);
    size_t stats_bytes = sizeof(hrm_hook_stats_t);
    if (g_owner_table) prefault_region(g_owner_table, owner_bytes);
    if (g_frame_net) prefault_region(g_frame_net, frame_bytes);
    if (g_ring) prefault_region(g_ring, ring_bytes);
    if (g_stats) prefault_region(g_stats, stats_bytes);
    g_stats->prefault_done = 1;
    return 1;
}

void hrm_alloc_hook_reset_aggregation_window(void) {
    if (!hook_should_run() || !g_frame_net) return;
    pthread_mutex_lock(&g_record_lock);
    memset(g_frame_net, 0, (size_t)HRM_FRAME_NET_SLOTS * sizeof(hrm_frame_net_t));
    if (g_stats) {
        g_stats->window_net_bytes = 0;
    }
    pthread_mutex_unlock(&g_record_lock);
}

static void refresh_top_sites(void) {
    if (!g_stats || !g_frame_net) return;
    memset(g_stats->top, 0, sizeof(g_stats->top));
    for (uint32_t i = 0; i < HRM_FRAME_NET_SLOTS; ++i) {
        hrm_frame_net_t *slot = &g_frame_net[i];
        if (!slot->valid || slot->net_bytes <= 0) continue;
        for (int t = 0; t < HRM_TOP_SITES; ++t) {
            if (!g_stats->top[t].owner_frame || slot->net_bytes > g_stats->top[t].net_bytes) {
                for (int s = HRM_TOP_SITES - 1; s > t; --s) g_stats->top[s] = g_stats->top[s - 1];
                g_stats->top[t].owner_frame = slot->owner_frame;
                g_stats->top[t].net_bytes = slot->net_bytes;
                g_stats->top[t].gross_alloc = slot->gross_alloc;
                g_stats->top[t].gross_free = slot->gross_free;
                g_stats->top[t].count = 1;
                break;
            }
        }
    }
}

int hrm_alloc_hook_flush_stats_json(const char *path) {
    if (!hook_should_run() || !g_stats || !path) return -1;
    pthread_mutex_lock(&g_record_lock);
    refresh_top_sites();
    FILE *f = fopen(path, "w");
    if (!f) {
        pthread_mutex_unlock(&g_record_lock);
        return -1;
    }
    fprintf(f, "{\"magic\":%" PRIu64 ",\"version\":%u,\"enabled\":%u,\"prefault_done\":%u,"
            "\"hook_table_start\":%" PRIu64 ",\"hook_table_end\":%" PRIu64 ","
            "\"hook_ring_start\":%" PRIu64 ",\"hook_ring_end\":%" PRIu64 ","
            "\"ring_drop_count\":%" PRIu64 ",\"lock_contention_drop_count\":%" PRIu64 ","
            "\"table_overflow_count\":%" PRIu64 ","
            "\"table_eviction_count\":%" PRIu64 ",\"unknown_free_bytes\":%" PRIu64 ","
            "\"unknown_free_unmeasured_count\":%" PRIu64 ",\"unknown_free_bytes_bounded\":%u,"
            "\"lost_owner_count\":%" PRIu64 ",\"window_net_bytes\":%" PRIu64 ",\"top_sites\":[",
            g_stats->magic, g_stats->version, g_stats->enabled, g_stats->prefault_done,
            g_stats->hook_table_start, g_stats->hook_table_end, g_stats->hook_ring_start,
            g_stats->hook_ring_end, g_stats->ring_drop_count, g_stats->lock_contention_drop_count,
            g_stats->table_overflow_count,
            g_stats->table_eviction_count, g_stats->unknown_free_bytes,
            g_stats->unknown_free_unmeasured_count, g_stats->unknown_free_bytes_bounded,
            g_stats->lost_owner_count, g_stats->window_net_bytes);
    for (int i = 0; i < HRM_TOP_SITES; ++i) {
        if (!g_stats->top[i].owner_frame) continue;
        fprintf(f, "%s{\"owner_frame\":\"0x%llx\",\"net_bytes\":%lld,\"gross_alloc\":%" PRIu64 ",\"gross_free\":%" PRIu64 "}",
                (i && g_stats->top[i - 1].owner_frame) ? "," : "",
                (unsigned long long)g_stats->top[i].owner_frame,
                (long long)g_stats->top[i].net_bytes,
                g_stats->top[i].gross_alloc, g_stats->top[i].gross_free);
    }
    fprintf(f, "]}\n");
    fclose(f);
    pthread_mutex_unlock(&g_record_lock);
    return 0;
}

void hrm_alloc_hook_note_positive_control(uint64_t size) {
    if (!hook_should_run() || !size) return;
    uintptr_t frames[HRM_FRAME_DEPTH];
    capture_frames(frames);
    record_alloc(11, (void *)(uintptr_t)1, size, frames);
    if (g_stats) g_stats->positive_control_hits++;
}
