from colorama import Fore, Style, init

init(autoreset=True)


class Logger:
    """Lightweight console-only logger used by experiment scripts."""

    def __init__(self, experiment_name="experiment"):
        self.experiment_name = experiment_name

    def info(self, message, color=None, bold=False):
        style_prefix = ""
        if bold:
            style_prefix += Style.BRIGHT
        if color:
            style_prefix += color
        print(f"{style_prefix}{message}{Style.RESET_ALL}")

    def section(self, title):
        border = "=" * 60
        self.info(f"\n{border}\n{title}\n{border}", color=Fore.CYAN, bold=True)

    def sub_section(self, title):
        self.info(f"\n>>> {title}", color=Fore.YELLOW, bold=True)

    def trade(self, action, price, wealth, ret):
        if action == "BUY":
            color = Fore.GREEN
        elif action == "SELL":
            color = Fore.RED
        else:
            color = Fore.WHITE

        message = f"[ACTION] {action:<4} @ {price:.2f} | Wealth: ${wealth:,.2f} ({ret:+.2f}%)"
        self.info(message, color=color, bold=True)


class NullLogger:
    """Drop-in logger that suppresses console output for batched experiments."""

    def __init__(self, experiment_name="experiment"):
        self.experiment_name = experiment_name

    def info(self, message, color=None, bold=False):
        return None

    def section(self, title):
        return None

    def sub_section(self, title):
        return None

    def trade(self, action, price, wealth, ret):
        return None
