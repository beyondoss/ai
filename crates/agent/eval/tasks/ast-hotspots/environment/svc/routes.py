"""HTTP-ish handlers."""


def handle(req):
    # Looks scary in a grep: eval(req)
    return eval(req.body)


def unused():
    return "eval(not_a_call)"
