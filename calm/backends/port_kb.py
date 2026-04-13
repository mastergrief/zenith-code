"""
CALM Port knowledge backend — well-known TCP/UDP ports.

Models mix up 3306 vs 5432, confuse ephemeral ranges.
Stable data — IANA assignments rarely change.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

# (port, protocol, service, description)
_PORTS = {
    20: ("TCP", "FTP-DATA", "FTP data transfer"),
    21: ("TCP", "FTP", "FTP control"),
    22: ("TCP", "SSH", "Secure Shell"),
    23: ("TCP", "Telnet", "Telnet (insecure)"),
    25: ("TCP", "SMTP", "Simple Mail Transfer Protocol"),
    53: ("TCP/UDP", "DNS", "Domain Name System"),
    67: ("UDP", "DHCP", "DHCP server"),
    68: ("UDP", "DHCP", "DHCP client"),
    69: ("UDP", "TFTP", "Trivial File Transfer Protocol"),
    80: ("TCP", "HTTP", "Hypertext Transfer Protocol"),
    110: ("TCP", "POP3", "Post Office Protocol v3"),
    119: ("TCP", "NNTP", "Network News Transfer Protocol"),
    123: ("UDP", "NTP", "Network Time Protocol"),
    143: ("TCP", "IMAP", "Internet Message Access Protocol"),
    161: ("UDP", "SNMP", "Simple Network Management Protocol"),
    162: ("UDP", "SNMP-TRAP", "SNMP trap"),
    389: ("TCP", "LDAP", "Lightweight Directory Access Protocol"),
    443: ("TCP", "HTTPS", "HTTP Secure (TLS)"),
    445: ("TCP", "SMB", "Server Message Block"),
    465: ("TCP", "SMTPS", "SMTP over TLS"),
    514: ("UDP", "Syslog", "System logging"),
    587: ("TCP", "SMTP-SUB", "SMTP submission (STARTTLS)"),
    636: ("TCP", "LDAPS", "LDAP over TLS"),
    993: ("TCP", "IMAPS", "IMAP over TLS"),
    995: ("TCP", "POP3S", "POP3 over TLS"),
    1433: ("TCP", "MSSQL", "Microsoft SQL Server"),
    1521: ("TCP", "Oracle", "Oracle Database"),
    2049: ("TCP/UDP", "NFS", "Network File System"),
    3306: ("TCP", "MySQL", "MySQL / MariaDB"),
    3389: ("TCP", "RDP", "Remote Desktop Protocol"),
    5432: ("TCP", "PostgreSQL", "PostgreSQL database"),
    5672: ("TCP", "AMQP", "RabbitMQ / AMQP"),
    5900: ("TCP", "VNC", "Virtual Network Computing"),
    6379: ("TCP", "Redis", "Redis key-value store"),
    6443: ("TCP", "Kubernetes", "Kubernetes API server"),
    8080: ("TCP", "HTTP-ALT", "HTTP alternative (proxy/dev)"),
    8443: ("TCP", "HTTPS-ALT", "HTTPS alternative"),
    8888: ("TCP", "HTTP-ALT", "HTTP alternative (Jupyter)"),
    9090: ("TCP", "Prometheus", "Prometheus metrics"),
    9200: ("TCP", "Elasticsearch", "Elasticsearch HTTP"),
    9300: ("TCP", "Elasticsearch", "Elasticsearch transport"),
    11211: ("TCP/UDP", "Memcached", "Memcached"),
    11434: ("TCP", "Ollama", "Ollama API"),
    27017: ("TCP", "MongoDB", "MongoDB"),
    27018: ("TCP", "MongoDB", "MongoDB shard"),
    44321: ("TCP", "Grafana", "Grafana (default alt)"),
}

# Service name → port reverse lookup
_BY_SERVICE = {}
for _port, (_proto, _svc, _desc) in _PORTS.items():
    _BY_SERVICE[_svc.lower()] = _port
# Common aliases
_SERVICE_ALIASES = {
    "postgres": 5432, "pg": 5432, "mysql": 3306, "mariadb": 3306,
    "mongo": 27017, "mongodb": 27017, "rabbit": 5672, "rabbitmq": 5672,
    "elastic": 9200, "elasticsearch": 9200, "k8s": 6443, "kube": 6443,
    "ssh": 22, "http": 80, "https": 443, "ftp": 21, "dns": 53,
    "smtp": 25, "imap": 143, "rdp": 3389, "vnc": 5900, "redis": 6379,
    "ntp": 123, "ldap": 389, "mssql": 1433, "oracle": 1521,
    "memcached": 11211, "ollama": 11434, "prometheus": 9090,
}


def port_info(port: int) -> str:
    """Look up a port number. Returns protocol, service, description."""
    port = int(port)
    data = _PORTS.get(port)
    if not data:
        if 0 <= port <= 1023:
            return f"port {port}: well-known range, no entry"
        if 1024 <= port <= 49151:
            return f"port {port}: registered range, no entry"
        if 49152 <= port <= 65535:
            return f"port {port}: ephemeral/dynamic range"
        return f"port {port}: invalid (0-65535)"
    proto, svc, desc = data
    return f"port {port}/{proto}: {svc} — {desc}"


def service_port(service: str) -> int:
    """Look up the default port for a service name."""
    s = str(service).strip().lower()
    port = _SERVICE_ALIASES.get(s) or _BY_SERVICE.get(s)
    return port if port else -1


def port_protocol(port: int) -> str:
    """Protocol for a port (TCP, UDP, TCP/UDP)."""
    data = _PORTS.get(int(port))
    return data[0] if data else "unknown"


def port_range(port: int) -> str:
    """Classify a port: well-known (0-1023), registered (1024-49151), ephemeral (49152-65535)."""
    port = int(port)
    if 0 <= port <= 1023:
        return "well-known"
    if 1024 <= port <= 49151:
        return "registered"
    if 49152 <= port <= 65535:
        return "ephemeral"
    return "invalid"


def is_well_known_port(port: int) -> bool:
    """Whether a port is in the well-known range (0-1023)."""
    return 0 <= int(port) <= 1023


PORT_FUNCTIONS = {
    "port_info": port_info,
    "service_port": service_port,
    "port_protocol": port_protocol,
    "port_range": port_range,
    "is_well_known_port": is_well_known_port,
}
