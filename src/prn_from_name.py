import re
from typing import Optional

PRN_REGEX = re.compile(r"(GPS [A-Z]+-\d+)+\s+\(PRN(?P<prn> \d+)\)")

def get_prn(name: str) -> Optional[int]:
    match = re.match(PRN_REGEX, name)
    if match is None:
        return None

    return int(match.group("prn"))