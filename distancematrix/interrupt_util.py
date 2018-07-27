import signal
from contextlib import contextmanager


@contextmanager
def interrupt_catcher():
    """
    A context that allows for gracefully terminating a calculation by catching interrupts
    and providing a method to check whether an interrupt has occurred.

    :return: None
    """
    interrupted = False

    def set_interrupted(signum, frame):
        nonlocal interrupted
        interrupted = True

    def is_interrupted():
        nonlocal interrupted
        return interrupted

    # Replace the current interrupt handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, set_interrupted)

    try:
        yield is_interrupted
    finally:
        # Restore original interrupt handler
        signal.signal(signal.SIGINT, original_sigint_handler)


if __name__ == "__main__":
    import time
    for i in range(5):
        print(i)
        time.sleep(1)

    print("-- Interrupts will now simply halt the loop --")
    with interrupt_catcher() as is_interrupted:
        for i in range(5):
            if is_interrupted():
                break
            print(i)
            time.sleep(1)
    print("-- Interrupts are back to normal --")
    for i in range(5):
        print(i)
        time.sleep(1)