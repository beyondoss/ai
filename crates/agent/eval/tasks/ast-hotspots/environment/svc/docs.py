"""Examples: eval(user_input) should not count. Neither should subprocess.Popen in this string."""


def describe():
    s = 'subprocess.Popen(["x"])'
    return s
