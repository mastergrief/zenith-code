"""
CALM Networking knowledge backend — OSI layers, protocols, common configs.

Models confuse layers, mix up TCP/UDP characteristics, hallucinate port assignments.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_OSI_LAYERS = {
    1: {"name": "Physical", "function": "Bit transmission over physical medium", "protocols": ["Ethernet (physical)", "USB", "Bluetooth", "Wi-Fi (physical)"], "devices": ["hub", "repeater", "cable", "NIC"], "pdu": "bits"},
    2: {"name": "Data Link", "function": "Reliable frame transfer, MAC addressing, error detection", "protocols": ["Ethernet", "Wi-Fi (802.11)", "PPP", "ARP"], "devices": ["switch", "bridge"], "pdu": "frames"},
    3: {"name": "Network", "function": "Logical addressing, routing between networks", "protocols": ["IP (IPv4/IPv6)", "ICMP", "OSPF", "BGP", "RIP"], "devices": ["router", "L3 switch"], "pdu": "packets"},
    4: {"name": "Transport", "function": "End-to-end delivery, flow control, error recovery", "protocols": ["TCP", "UDP", "QUIC", "SCTP"], "devices": ["firewall", "load balancer"], "pdu": "segments/datagrams"},
    5: {"name": "Session", "function": "Session establishment, maintenance, teardown", "protocols": ["NetBIOS", "RPC", "PPTP"], "devices": [], "pdu": "data"},
    6: {"name": "Presentation", "function": "Data format translation, encryption, compression", "protocols": ["SSL/TLS", "JPEG", "MPEG", "ASCII/Unicode"], "devices": [], "pdu": "data"},
    7: {"name": "Application", "function": "User-facing services and interfaces", "protocols": ["HTTP/HTTPS", "DNS", "FTP", "SMTP", "SSH", "DHCP", "SNMP"], "devices": ["proxy", "WAF"], "pdu": "data"},
}

_PROTOCOLS = {
    "TCP": {"layer": 4, "connection": "connection-oriented", "reliable": True, "ordered": True, "header_size": "20-60 bytes", "use": "web, email, file transfer", "flow_control": "sliding window", "handshake": "3-way (SYN, SYN-ACK, ACK)"},
    "UDP": {"layer": 4, "connection": "connectionless", "reliable": False, "ordered": False, "header_size": "8 bytes", "use": "DNS, video streaming, gaming, VoIP", "flow_control": "none", "handshake": "none"},
    "QUIC": {"layer": 4, "connection": "connection-oriented", "reliable": True, "ordered": True, "header_size": "variable", "use": "HTTP/3, modern web", "built_on": "UDP", "features": "0-RTT, multiplexing, built-in TLS 1.3"},
    "HTTP": {"layer": 7, "versions": ["1.0", "1.1", "2", "3"], "default_port": 80, "methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]},
    "HTTPS": {"layer": 7, "default_port": 443, "encryption": "TLS 1.2/1.3", "certificate": "X.509"},
    "DNS": {"layer": 7, "default_port": 53, "protocol": "UDP (queries), TCP (zone transfers)", "record_types": ["A", "AAAA", "CNAME", "MX", "NS", "TXT", "SOA", "SRV"]},
    "SSH": {"layer": 7, "default_port": 22, "encryption": "AES-256, ChaCha20", "auth": ["password", "public key", "certificate"]},
    "FTP": {"layer": 7, "default_port": "20 (data), 21 (control)", "modes": ["active", "passive"], "secure": False, "alternatives": ["SFTP", "SCP", "FTPS"]},
    "SMTP": {"layer": 7, "default_port": "25 (relay), 587 (submission)", "encryption": "STARTTLS", "use": "sending email"},
    "IMAP": {"layer": 7, "default_port": 993, "encryption": "TLS", "use": "reading email (server-side)"},
    "DHCP": {"layer": 7, "default_port": "67 (server), 68 (client)", "protocol": "UDP", "process": "DORA (Discover, Offer, Request, Acknowledge)"},
    "ARP": {"layer": 2, "function": "IP → MAC address resolution", "scope": "local network only", "vulnerability": "ARP spoofing"},
    "ICMP": {"layer": 3, "function": "Network diagnostics", "tools": ["ping", "traceroute"], "not_for": "data transfer"},
    "BGP": {"layer": 3, "default_port": 179, "type": "path-vector", "use": "inter-AS routing (internet backbone)", "protocol": "TCP"},
    "OSPF": {"layer": 3, "type": "link-state", "use": "intra-AS routing", "metric": "cost (bandwidth-based)"},
}

_TCP_VS_UDP = {
    "TCP": {"reliable": True, "ordered": True, "connection": True, "overhead": "high", "speed": "slower", "use_when": "data integrity matters (web, email, files)"},
    "UDP": {"reliable": False, "ordered": False, "connection": False, "overhead": "low", "speed": "faster", "use_when": "speed matters, some loss OK (streaming, gaming, DNS)"},
}

_DNS_RECORD_TYPES = {
    "A": {"description": "Maps hostname to IPv4 address", "example": "example.com → 93.184.216.34"},
    "AAAA": {"description": "Maps hostname to IPv6 address", "example": "example.com → 2606:2800:220:1:..."},
    "CNAME": {"description": "Alias for another hostname", "example": "www.example.com → example.com", "gotcha": "Cannot coexist with other records at same name"},
    "MX": {"description": "Mail server for domain", "example": "example.com → mail.example.com (priority 10)"},
    "NS": {"description": "Authoritative nameserver for domain", "example": "example.com → ns1.example.com"},
    "TXT": {"description": "Arbitrary text (SPF, DKIM, verification)", "example": "v=spf1 include:_spf.google.com ~all"},
    "SOA": {"description": "Start of Authority — primary NS, admin email, serial", "example": "ns1.example.com admin.example.com 2024010101"},
    "SRV": {"description": "Service location (host, port, priority, weight)", "example": "_sip._tcp.example.com → sipserver.example.com:5060"},
    "PTR": {"description": "Reverse DNS — IP to hostname", "example": "34.216.184.93.in-addr.arpa → example.com"},
    "CAA": {"description": "Certificate Authority Authorization", "example": "0 issue letsencrypt.org"},
}


def osi_layer(number: int) -> dict:
    """Get OSI model layer details by number (1-7)."""
    entry = _OSI_LAYERS.get(int(number))
    if not entry:
        return {"error": f"Invalid layer: {number} (valid: 1-7)"}
    return {"layer": int(number), **entry}


def protocol_info(name: str) -> dict:
    """Get details about a network protocol."""
    key = str(name).upper().strip()
    entry = _PROTOCOLS.get(key)
    if not entry:
        return {"error": f"Unknown protocol: {name}", "valid": sorted(_PROTOCOLS.keys())}
    return {"protocol": key, **entry}


def tcp_vs_udp() -> dict:
    """Compare TCP and UDP."""
    return _TCP_VS_UDP


def dns_record(record_type: str) -> dict:
    """Get DNS record type details."""
    key = str(record_type).upper().strip()
    entry = _DNS_RECORD_TYPES.get(key)
    if not entry:
        return {"error": f"Unknown type: {record_type}", "valid": list(_DNS_RECORD_TYPES.keys())}
    return {"type": key, **entry}


def which_layer(protocol: str) -> int:
    """Which OSI layer does a protocol operate at?"""
    key = str(protocol).upper().strip()
    entry = _PROTOCOLS.get(key)
    if not entry:
        return -1
    return entry.get("layer", -1)


def list_layer_protocols(layer: int) -> list[str]:
    """List all protocols at a given OSI layer."""
    l = int(layer)
    return sorted(k for k, v in _PROTOCOLS.items() if v.get("layer") == l)


NETWORKING_FUNCTIONS = {
    "osi_layer": osi_layer,
    "protocol_info": protocol_info,
    "tcp_vs_udp": tcp_vs_udp,
    "dns_record": dns_record,
    "which_layer": which_layer,
    "list_layer_protocols": list_layer_protocols,
}

NETWORKING_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:OSI\s+)?layer\s+(\d)', 'osi_layer({0})'),
    (r'(?:what is|explain|info about)\s+(TCP|UDP|HTTP|HTTPS|DNS|SSH|FTP|SMTP|DHCP|ARP|ICMP|BGP|OSPF|QUIC)', 'protocol_info("{0}")'),
    (r'(?:difference between|compare|vs)\s+TCP\s+(?:and|vs)\s+UDP', 'tcp_vs_udp()'),
    (r'(?:what is|explain)\s+(?:a\s+)?(?:DNS\s+)?(A|AAAA|CNAME|MX|NS|TXT|SOA|SRV|PTR|CAA)\s+record', 'dns_record("{0}")'),
    (r'(?:which|what)\s+(?:OSI\s+)?layer\s+(?:is|does)\s+(\w+)', 'which_layer("{0}")'),
]
