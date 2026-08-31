import os
import platform
import subprocess
import sys

import typer


app = typer.Typer(
    name="hg-mc-auto",
    help="Hg-MC-Auto: automated mercury isotope data analysis.",
    no_args_is_help=False,
)

APP_LANGUAGE = "zh"
LANG_TEXT = {
    "zh": {
        "welcome": "欢迎使用 Hg-MC-Auto",
        "subtitle": "汞同位素分析自动化工作流",
        "menu_title": "请选择任务",
        "option_export": "1. 自动导出同位素数据",
        "option_calculate": "2. 导出仪器参数并计算同位素分馏值",
        "option_empirical": "3. 使用经验模型进行分类",
        "option_predict": "4. 使用机器学习模型进行分类",
        "option_train": "5. 训练自定义机器学习专家模型",
        "option_language": "8. 切换语言（中文 / English）",
        "option_exit": "0. 退出",
        "prompt_choice": "请输入选择 (0-8): ",
        "goodbye": "感谢使用 Hg-MC-Auto，再见！",
        "invalid": "无效输入，请输入 0-8 之间的数字。",
        "language_prompt": "请选择界面语言: 1=中文 / 2=English",
        "language_set": "界面语言已切换为中文。",
        "language_set_en": "Interface language set to English.",
        "cancelled": "已取消。",
        "confirm_train": "是否继续执行两步训练流程?",
    },
    "en": {
        "welcome": "Welcome to Hg-MC-Auto",
        "subtitle": "Mercury isotope analysis automation pipeline",
        "menu_title": "Please select a task",
        "option_export": "1. Automatically export isotope data",
        "option_calculate": "2. Export instrument parameters and calculate isotope fractionation",
        "option_empirical": "3. Classify data using the empirical model",
        "option_predict": "4. Classify data using the machine learning model",
        "option_train": "5. Train your custom machine learning expert model",
        "option_language": "8. Switch language (中文 / English)",
        "option_exit": "0. Exit",
        "prompt_choice": "Enter your choice (0-8): ",
        "goodbye": "Thank you for using Hg-MC-Auto. Goodbye!",
        "invalid": "Invalid choice. Please enter a number between 0 and 8.",
        "language_prompt": "Please choose the interface language: 1=中文 / 2=English",
        "language_set": "界面语言已切换为中文。",
        "language_set_en": "Interface language set to English.",
        "cancelled": "Operation cancelled.",
        "confirm_train": "Run both machine learning training stages?",
    },
}

TASKS = {
    "export": ("1_Automatic_export.py", "Automatically export isotope data"),
    "calculate": (
        "2_Automatic_calculation.py",
        "Merge isotope data and calculate fractionation values",
    ),
    "empirical": ("3_Empirical_model.py", "Classify data with the empirical model"),
    "predict": ("4_ML_Predict.py", "Classify data with a machine learning model"),
    "train-external": ("5.Exter_ML_train.py", "Train the external ML model"),
    "train-internal": ("6.Inter_ML_train.py", "Train the internal ML model"),
}


def t(key):
    return LANG_TEXT.get(APP_LANGUAGE, LANG_TEXT["zh"]).get(key, key)


def set_language(language):
    global APP_LANGUAGE
    normalized = language.lower()
    if normalized in {"zh", "cn", "zh-cn", "中文"}:
        APP_LANGUAGE = "zh"
        return "zh"
    if normalized in {"en", "eng", "english"}:
        APP_LANGUAGE = "en"
        return "en"
    raise ValueError("Language must be 'zh' or 'en'.")


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the program header in a polished terminal layout."""
    typer.secho("=" * 78, fg=typer.colors.CYAN, bold=True)
    typer.secho(f"{t('welcome').center(78)}", fg=typer.colors.BRIGHT_CYAN, bold=True)
    typer.secho(f"{t('subtitle').center(78)}", fg=typer.colors.WHITE)
    typer.secho("=" * 78, fg=typer.colors.CYAN, bold=True)


def print_menu():
    """Print the main menu options."""
    print(f"\n{t('menu_title')}:")
    print(t('option_export'))
    print(t('option_calculate'))
    print(t('option_empirical'))
    print(t('option_predict'))
    print(t('option_train'))
    print(t('option_language'))
    print(t('option_exit'))
    print("-" * 60)

def get_script_path(script_name):
    """Get the full path of a script in the same directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    return script_path

def run_script(script_name):
    """Run a Python script in the same directory."""
    script_path = get_script_path(script_name)

    if not os.path.exists(script_path):
        typer.secho(f"Error: script '{script_name}' was not found.", fg=typer.colors.RED, bold=True)
        typer.secho(f"Expected path: {script_path}", fg=typer.colors.RED)
        return False

    try:
        typer.secho(f"\n[STEP] Running {script_name}...", fg=typer.colors.GREEN, bold=True)
        typer.secho("=" * 48, fg=typer.colors.MAGENTA)
        subprocess.run([sys.executable, script_path], check=True)
        typer.secho("=" * 48, fg=typer.colors.MAGENTA)
        typer.secho(f"[OK] {script_name} completed successfully.", fg=typer.colors.GREEN, bold=True)
        return True
    except subprocess.CalledProcessError as e:
        typer.secho(f"[ERROR] Failed to run {script_name}: {e}", fg=typer.colors.RED, bold=True)
        return False
    except FileNotFoundError:
        typer.secho("[ERROR] Python interpreter not found.", fg=typer.colors.RED, bold=True)
        return False
    except Exception as e:
        typer.secho(f"[ERROR] Unexpected error running {script_name}: {e}", fg=typer.colors.RED, bold=True)
        return False

def wait_for_enter():
    """Wait for user to press Enter"""
    input("\nPress Enter to continue...")

def run_task(task_name):
    """Run a registered task and return whether it succeeded."""
    script_name, description = TASKS[task_name]
    print(f"\nYou selected: {description}")
    return run_script(script_name)


def train_all():
    """Run both training stages in order."""
    if not run_task("train-external"):
        print("Cannot proceed with internal model training.")
        return False
    return run_task("train-internal")


def select_language_interactive():
    """Prompt the user to choose the interface language."""
    while True:
        choice = input(f"\n{t('language_prompt')}: ").strip().lower()
        if choice in {'1', 'zh', 'cn', '中文'}:
            set_language('zh')
            typer.secho(t('language_set'), fg=typer.colors.CYAN, bold=True)
            return
        if choice in {'2', 'en', 'english'}:
            set_language('en')
            typer.secho(t('language_set_en'), fg=typer.colors.CYAN, bold=True)
            return
        typer.secho("Invalid selection. Please choose 1 or 2.", fg=typer.colors.YELLOW)


def main():
    """Main program loop."""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input(f"\n{t('prompt_choice')}").strip()

        if choice == '0':
            typer.secho(f"\n{t('goodbye')}", fg=typer.colors.BRIGHT_GREEN, bold=True)
            break

        elif choice == '1':
            typer.secho("\n[SELECTED] Automatically export isotope data", fg=typer.colors.BRIGHT_CYAN, bold=True)
            run_task("export")
            wait_for_enter()

        elif choice == '2':
            typer.secho("\n[SELECTED] Export instrument parameters and calculate isotope fractionation values", fg=typer.colors.BRIGHT_CYAN, bold=True)
            run_task("calculate")
            wait_for_enter()

        elif choice == '3':
            typer.secho("\n[SELECTED] Classify data using an empirical model", fg=typer.colors.BRIGHT_CYAN, bold=True)
            run_task("empirical")
            wait_for_enter()

        elif choice == '4':
            typer.secho("\n[SELECTED] Classify data using a machine learning model", fg=typer.colors.BRIGHT_CYAN, bold=True)
            run_task("predict")
            wait_for_enter()

        elif choice == '5':
            typer.secho("\n[SELECTED] Train your own machine learning expert model", fg=typer.colors.BRIGHT_CYAN, bold=True)
            typer.secho("This will run both training scripts sequentially:", fg=typer.colors.WHITE)
            typer.secho("1. 5.Exter_ML_train.py", fg=typer.colors.WHITE)
            typer.secho("2. 6.Inter_ML_train.py", fg=typer.colors.WHITE)

            confirm = input(f"\n{t('confirm_train')} (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                if train_all():
                    typer.secho("\n[OK] Machine learning training completed successfully!", fg=typer.colors.GREEN, bold=True)
                else:
                    typer.secho("\n[WARN] Machine learning training had some issues.", fg=typer.colors.YELLOW, bold=True)
            else:
                typer.secho(t('cancelled'), fg=typer.colors.YELLOW)

            wait_for_enter()

        elif choice == '8':
            select_language_interactive()
            wait_for_enter()

        else:
            typer.secho(f"\n{t('invalid')}", fg=typer.colors.RED, bold=True)
            wait_for_enter()


def command_for(task_name):
    """Create a Typer command for one of the legacy interactive scripts."""
    def command():
        if not run_task(task_name):
            raise typer.Exit(code=1)

    command.__name__ = task_name.replace("-", "_")
    command.__doc__ = TASKS[task_name][1]
    return command


for task_name in (
    "export",
    "calculate",
    "empirical",
    "predict",
    "train-external",
    "train-internal",
):
    app.command(task_name)(command_for(task_name))


@app.command("train")
def train(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Start both training stages without confirmation.",
    ),
):
    """Train the external and internal ML models in order."""
    if not yes and not typer.confirm("Run both machine learning training stages?"):
        typer.echo("Operation cancelled.")
        raise typer.Exit()
    if not train_all():
        raise typer.Exit(code=1)


@app.command("language")
def language_command(
    lang: str = typer.Option(
        None,
        "--lang",
        "-l",
        help="Set the UI language. Options: zh, en",
    ),
):
    """Set the CLI interface language to Chinese or English."""
    if lang is None:
        typer.echo("Select language: 1=中文 / 2=English")
        choice = typer.prompt("Choice", default="1")
        lang = "zh" if choice in {"1", "zh", "中文", "cn"} else "en"
    try:
        set_language(lang)
    except ValueError:
        typer.echo("Unsupported language. Use 'zh' or 'en'.")
        raise typer.Exit(code=1)
    typer.echo("UI language set to: " + APP_LANGUAGE)


@app.command("platform")
def platform_support():
    """Show supported operating systems and the current platform status."""
    current = platform.system()
    supported = {
        "Windows": current == "Windows",
        "Linux": current == "Linux",
        "macOS": current == "Darwin",
    }

    typer.echo("Supported platforms: Windows, Linux, macOS")
    typer.echo(f"Current platform: {current or 'unknown'}")
    typer.echo("Desktop automation status: ")

    for name, is_current in supported.items():
        if is_current:
            typer.echo(f"- {name}: current")
        else:
            typer.echo(f"- {name}: supported")

    if current in {"Windows", "Linux", "Darwin"}:
        typer.echo("Status: supported by the project runtime.")
    else:
        typer.echo("Status: unsupported operating system.")

    if current == "Linux":
        typer.echo("Note: GUI automation requires a desktop session and a valid DISPLAY/X11 environment.")
    elif current == "Darwin":
        typer.echo("Note: GUI automation is supported on macOS desktop sessions.")
    elif current == "Windows":
        typer.echo("Note: GUI automation is supported on Windows desktop sessions.")


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            main()
        else:
            app()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        print("Please check your installation and try again.")