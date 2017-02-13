import re

match = re.search("\w+ idea", "fabulous idea")
match = re.search("@\w+ try \w+", "@vicegd try terminator and con air")
match = re.search("(.*) (re-)?watch (.*) fabulous (.*)", "deciding to re-watch breaking bad is a fabulous idea", flags=re.I)
if match:
    print("OK")
else:
    print("NO")