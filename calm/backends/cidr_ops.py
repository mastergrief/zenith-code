"""
CALM CIDR/Subnet backend — IP calculations, subnet masks, ranges.

Models can't do subnet math. Pure stdlib ipaddress module.
"""

from __future__ import annotations

import ipaddress


def cidr_info(cidr: str) -> str:
    """Full info for a CIDR block (e.g., '192.168.1.0/24')."""
    try:
        net = ipaddress.ip_network(str(cidr), strict=False)
        return (f"network={net.network_address}, broadcast={net.broadcast_address}, "
                f"mask={net.netmask}, hosts={net.num_addresses - 2}, "
                f"range={net[1]}-{net[-2]}")
    except ValueError as e:
        return f"invalid CIDR: {e}"


def subnet_mask(prefix_len: int) -> str:
    """Subnet mask for a prefix length (e.g., 24 → 255.255.255.0)."""
    try:
        net = ipaddress.ip_network(f"0.0.0.0/{int(prefix_len)}", strict=False)
        return str(net.netmask)
    except ValueError as e:
        return f"invalid prefix: {e}"


def hosts_in_subnet(prefix_len: int) -> int:
    """Number of usable hosts in a subnet (excludes network + broadcast)."""
    p = int(prefix_len)
    if p < 0 or p > 32:
        return -1
    return max(0, 2 ** (32 - p) - 2)


def ip_in_subnet(ip: str, cidr: str) -> bool:
    """Check if an IP address is in a CIDR range."""
    try:
        return ipaddress.ip_address(str(ip)) in ipaddress.ip_network(str(cidr), strict=False)
    except ValueError:
        return False


def subnets_overlap(cidr1: str, cidr2: str) -> bool:
    """Check if two CIDR blocks overlap."""
    try:
        net1 = ipaddress.ip_network(str(cidr1), strict=False)
        net2 = ipaddress.ip_network(str(cidr2), strict=False)
        return net1.overlaps(net2)
    except ValueError:
        return False


def split_subnet(cidr: str, new_prefix: int) -> list:
    """Split a CIDR block into smaller subnets."""
    try:
        net = ipaddress.ip_network(str(cidr), strict=False)
        return [str(s) for s in net.subnets(new_prefix=int(new_prefix))]
    except ValueError as e:
        return [f"error: {e}"]


def ip_version(ip: str) -> int:
    """Detect IP version (4 or 6)."""
    try:
        return ipaddress.ip_address(str(ip)).version
    except ValueError:
        return -1


def is_private_ip(ip: str) -> bool:
    """Check if an IP is in a private range (RFC 1918)."""
    try:
        return ipaddress.ip_address(str(ip)).is_private
    except ValueError:
        return False


CIDR_FUNCTIONS = {
    "cidr_info": cidr_info,
    "subnet_mask": subnet_mask,
    "hosts_in_subnet": hosts_in_subnet,
    "ip_in_subnet": ip_in_subnet,
    "subnets_overlap": subnets_overlap,
    "split_subnet": split_subnet,
    "ip_version": ip_version,
    "is_private_ip": is_private_ip,
}

CIDR_NL_PATTERNS = [
    (r'(?:how many|number of)\s+hosts?\s+in\s+(?:a\s+)?/(\d+)', 'hosts_in_subnet({0})'),
    (r'subnet mask\s+(?:for|of)\s+/(\d+)', 'subnet_mask({0})'),
    (r'(?:is\s+)?([\d.]+)\s+(?:in|inside|within)\s+([\d./]+)', 'ip_in_subnet("{0}", "{1}")'),
    (r'(?:is\s+)?([\d.]+)\s+(?:a\s+)?private', 'is_private_ip("{0}")'),
    (r'(?:do\s+)?([\d./]+)\s+and\s+([\d./]+)\s+overlap', 'subnets_overlap("{0}", "{1}")'),
]
