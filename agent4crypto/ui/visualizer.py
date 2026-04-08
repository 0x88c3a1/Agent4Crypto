import random
import sys
import time

from colorama import Fore, Style, init

init(autoreset=True)


class SystemVisualizer:
    def __init__(self):
        self.typing_speed = 0.0
        self.line_delay = 0.0

    def _render_banner_line(self, text, color=Fore.CYAN):
        """Render a banner line with a lightweight typewriter effect."""
        sys.stdout.write(color)
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(self.typing_speed + random.uniform(0, 0.001))
        print(Style.RESET_ALL)
        time.sleep(self.line_delay)

    def _banner_cyberpunk(self):
        banner_art = r"""
            ___                   __    __ __   ______                 __
           /   | ____ ____  ____  / /_  / // /  / ____/________  ______  / /_____
          / /| |/ __ `/ _ \/ __ \/ __/ / // /_ / /   / ___/ __ \/ __ \/ __/ __ \
         / ___ / /_/ /  __/ / / / /_  /__  __/ / /___/ /  / /_/ / /_/ / /_/ /_/ /
        /_/  |_\__, /\___/_/ /_/\__/    /_/    \____/_/   \__, / .___/\__/\____/
               /____/                                    /____/_/
        """

        lines = banner_art.strip("\n").split("\n")
        print("\n")
        for line in lines:
            self._render_banner_line(line)
        print("\n")

    def boot_sequence(self):
        self._banner_cyberpunk()


if __name__ == "__main__":
    SystemVisualizer().boot_sequence()
