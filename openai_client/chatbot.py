import sys
from typing import List, Dict
from utils import create_llm_client, LLMClient
from tools import get_weather, TOOLS_CONFIG
import time
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, FloatPrompt
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import box
import random
from datetime import datetime

console = Console()

PROVIDERS = ["openai", "deepseek", "dashscope", "zhipu", "siliconflow", "ollama"]

def animated_text(text: str, delay: float = 0.03):
    """动画效果显示文本"""
    for char in text:
        console.print(char, end='', style='bold blue')
        time.sleep(delay)
    console.print()

def select_provider() -> str:
    """让用户选择 provider"""
    console.print(Panel(
        "\n".join([f"[cyan]{i+1}.[/cyan] {provider}" for i, provider in enumerate(PROVIDERS)]),
        title="Available Providers",
        border_style="blue"
    ))
    while True:
        choice = Prompt.ask(
            "Select provider",
            choices=[str(i+1) for i in range(len(PROVIDERS))],
            default="1"
        )
        return PROVIDERS[int(choice)-1]

def select_model(client: LLMClient) -> str:
    """让用户从可用模型列表中选择"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Fetching available models...", total=None)
        models = client.list_models()
    
    if not models:
        console.print("[yellow]No models found, using default model[/yellow]")
        return client.model
    
    console.print(Panel(
        "\n".join([f"[cyan]{i+1}.[/cyan] {model}" for i, model in enumerate(models)]) + 
        "\n\n[italic]You can enter a number or directly type the model name[/italic]",
        title="Available Models",
        border_style="blue"
    ))
    
    while True:
        choice = Prompt.ask(
            "Select model (number or model name)",
            default="1"
        )
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice)-1]
        elif choice in models:
            return choice
        else:
            console.print("[red]Invalid choice. Please enter a valid number or model name.[/red]")

def get_system_prompt() -> str:
    """获取用户自定义的系统提示词"""
    default_prompt = ""
    return Prompt.ask(
        "\n[bold]Enter system prompt[/bold]",
        default=default_prompt
    )

def get_temperature() -> float:
    """获取用户设定的temperature值"""
    while True:
        try:
            temp = FloatPrompt.ask(
                "\n[bold]Enter temperature[/bold] (0.0-1.0)",
                default=0.7
            )
            if 0.0 <= temp <= 1.0:
                return temp
            else:
                console.print("[red]Temperature must be between 0.0 and 1.0[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")

def display_welcome():
    title = """
    🤖 AI Chat Assistant 🤖
    """
    welcome_text = "Welcome to an enhanced chat experience!"
    
    console.print(Panel(
        Text(title, justify="center", style="bold blue"),
        box=box.DOUBLE,
        border_style="blue",
        padding=(1, 2)
    ))
    
    animated_text(welcome_text)
    console.print("\n💡 Type [bold cyan]'exit'[/bold cyan] or [bold cyan]'quit'[/bold cyan] to end the conversation\n")

def get_user_input() -> str:
    prompt_symbols = ["💭", "💬", "🗨️", "💡"]
    return console.input(f"\n{random.choice(prompt_symbols)} [bold green]You:[/bold green] ")

def display_thinking_animation(live, text="", token_speed=0):
    """显示思考动画和token速度"""
    return f"{text}\n[cyan]Speed: {token_speed:.1f} tokens/s[/cyan]"

def format_assistant_response(text: str, token_count: int, total_tokens: int, token_speed: float = 0) -> Panel:
    """格式化助手响应，包含token统计和速度"""
    return Panel(
        Text.from_markup(
            f"{text}\n\n" +
            f"[dim]Response tokens: {token_count}[/dim]\n" +
            f"[dim]Total tokens: {total_tokens}[/dim]\n" +
            f"[cyan]Speed: {token_speed:.1f} tokens/s[/cyan]"
        ),
        border_style="purple",
        box=box.ROUNDED,
        title="[bold purple]Assistant[/bold purple]",
        title_align="left",
        padding=(1, 2)
    )

def main():
    display_welcome()
    
    # 配置聊天环境
    provider = select_provider()
    client = create_llm_client(provider)
    
    # 选择模型
    selected_model = select_model(client)
    client.model = selected_model
    
    # 设置temperature
    temperature = get_temperature()
    
    # 设置系统提示词
    system_prompt = get_system_prompt()
    
    console.print(Panel(
        f"[cyan]Provider:[/cyan] {provider}\n"
        f"[cyan]Model:[/cyan] {selected_model}\n"
        f"[cyan]Temperature:[/cyan] {temperature}\n",
        title="Chat Configuration",
        border_style="green"
    ))
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    
    total_tokens = 0
    
    while True:
        user_input = get_user_input()
        
        if user_input.lower() in ['exit', 'quit']:
            console.print(Panel(
                "👋 Thanks for chatting! Have a great day!",
                style="bold blue",
                box=box.ROUNDED
            ))
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # 收集完整响应
        full_response = ""
        reasoning_content = ""
        start_time = time.time()
        first_token_time = None
        first_token_latency = None
        token_count = 0
        
        with Live(refresh_per_second=10) as live:
            try:
                for chunk_type, chunk in client.stream_chat_completion(
                    messages=messages,
                    temperature=temperature
                ):
                    # 记录第一个token的时间
                    if first_token_time is None and (chunk_type == 'content' or chunk_type == 'reasoning'):
                        first_token_time = time.time()
                        first_token_latency = first_token_time - start_time

                    elapsed = time.time() - start_time
                    token_speed = token_count / elapsed if elapsed > 0 else 0
                    
                    # 准备延迟信息
                    latency_info = f"[yellow]First token latency: {first_token_latency:.2f}s[/yellow]\n" if first_token_latency is not None else ""
                    
                    if chunk_type == 'reasoning':
                        reasoning_content += chunk
                        live.update(Panel(
                            Text.from_markup(
                                f"{reasoning_content}\n\n{latency_info}"
                            ),
                            title="💭 Reasoning",
                            border_style="yellow",
                            box=box.ROUNDED
                        ))
                    elif chunk_type == 'content':
                        if reasoning_content:
                            console.print(Panel(
                                Text.from_markup(
                                    f"{reasoning_content}\n\n{latency_info}"
                                ),
                                title="💭 Reasoning",
                                border_style="yellow",
                                box=box.ROUNDED
                            ))
                            reasoning_content = ""
                            
                        full_response += chunk
                        token_count += 1
                        
                        live.update(Panel(
                            Text.from_markup(
                                f"{full_response}\n\n" +
                                f"{latency_info}" +
                                f"[dim]Response tokens: {token_count}[/dim]\n" +
                                f"[dim]Total tokens: {total_tokens + token_count}[/dim]\n" +
                                f"[cyan]Speed: {token_speed:.1f} tokens/s[/cyan]\n" +
                                f"[dim]Time elapsed: {elapsed:.1f}s[/dim]"
                            ),
                            border_style="purple",
                            box=box.ROUNDED,
                            title="[bold purple]Assistant[/bold purple]",
                            title_align="left",
                            padding=(1, 2)
                        ))

            except Exception as e:
                console.print(Panel(
                    f"❌ Error: {str(e)}",
                    style="bold red",
                    box=box.ROUNDED
                ))
                continue
        
        console.print("─" * console.width, style="dim")
        total_tokens += token_count
        messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(Panel(
            "👋 Chat ended by user. Goodbye!",
            style="bold blue",
            box=box.ROUNDED
        ))
        sys.exit(0)
