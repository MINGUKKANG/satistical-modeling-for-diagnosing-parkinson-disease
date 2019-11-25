def print_arg(args):
    for arg in vars(args):
        log_string = arg
        log_string += "." * (100 - len(arg) - len(str(getattr(args, arg))))
        log_string += str(getattr(args, arg))
        print(log_string)