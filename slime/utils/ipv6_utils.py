"""Utilities for handling IPv6 addresses in network operations."""


def is_ipv6(address: str) -> bool:
    """Check if the given address is an IPv6 address.

    Args:
        address: An IP address string (may or may not have brackets).

    Returns:
        True if the address appears to be IPv6, False otherwise.
    """
    stripped = address.strip("[]")
    return ":" in stripped


def format_ipv6_address(address: str) -> str:
    """Format an IPv6 address with brackets for URL compatibility.

    IPv6 addresses must be wrapped in brackets when used in URLs,
    e.g., http://[2001:db8::1]:8080/

    Args:
        address: An IP address string (IPv4 or IPv6).

    Returns:
        The address with brackets added if it's IPv6, or unchanged if IPv4.
    """
    if is_ipv6(address) and not address.startswith("["):
        return f"[{address}]"
    return address


def format_tcp_init_method(address: str, port: int) -> str:
    """Format a TCP init method URL for PyTorch distributed.

    Creates a proper tcp:// URL that handles both IPv4 and IPv6 addresses.

    Args:
        address: The master IP address (IPv4 or IPv6).
        port: The port number.

    Returns:
        A properly formatted tcp:// URL.
    """
    formatted_address = format_ipv6_address(address)
    return f"tcp://{formatted_address}:{port}"
