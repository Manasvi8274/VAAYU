"""
Permanent safety rules, kept in guardian/ (outside assistant/) so Vaayu's
self-upgrade can never edit them - the self-upgrade gate refuses any change
touching PROTECTED_PATHS. assistant/ tools import from here; nothing here
imports from assistant/.
"""

import re

# Paths (relative to the repo root) a self-upgrade may never change.
PROTECTED_PATHS = ("guardian/",)

# Payment apps / sites are blocked entirely - not just transactions, even
# opening them (explicit request: "No payment access to Vaayu ever").
_PAYMENT = re.compile(
    r"\b(?:paytm|phone ?pe|g ?pay|google ?pay|bhim|upi|paypal|razorpay|cred|mobikwik|freecharge|amazon ?pay|"
    r"net ?banking|onlinesbi|sbi ?yono|yono|hdfc ?bank|icici ?bank|axis ?bank|kotak|pnb|bank ?of ?baroda|"
    r"canara ?bank|idfc|indusind|yes ?bank|netbanking)\b",
    re.I,
)


def is_payment_target(name_or_url: str) -> bool:
    """True for a payment/banking app name or website."""
    return bool(_PAYMENT.search(re.sub(r"[._/-]", " ", name_or_url or "")))


def check_not_payment(name_or_url: str) -> None:
    if is_payment_target(name_or_url):
        raise PermissionError("Payment and banking apps are blocked for Vaayu permanently.")
