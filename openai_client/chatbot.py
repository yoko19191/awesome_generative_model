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
import json

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
    """让用户从可用模型列表中选择或输入自定义模型名称"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Fetching available models...", total=None)
        models = client.list_models()
    
    if not models:
        custom_model = Prompt.ask(
            "Enter model name",
            default=client.model
        )
        return custom_model
    
    console.print(Panel(
        "\n".join([f"[cyan]{i+1}.[/cyan] {model}" for i, model in enumerate(models)]) + 
        "\n\n[italic]You can enter a number, model name from the list, or any custom model name[/italic]",
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
        else:
            return choice  # 返回用户输入的任何模型名称

def get_system_prompt() -> str:
    """获取用户自定义的系统提示词"""
    default_prompt = "You are a helpful assistant. You can use function calling to get external knowledge."
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

def enable_tools() -> bool:
    """让用户选择是否启用 tools 功能"""
    return Prompt.ask(
        "\n[bold]Enable tools (weather, etc.)?[/bold] (y/n)",
        choices=["y", "n"],
        default="y"
    ).lower() == "y"

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
    
    # 选择是否启用 tools
    tools_enabled = enable_tools()
    
    console.print(Panel(
        f"[cyan]Provider:[/cyan] {provider}\n"
        f"[cyan]Model:[/cyan] {selected_model}\n"
        f"[cyan]Temperature:[/cyan] {temperature}\n"
        f"[cyan]Tools enabled:[/cyan] {'Yes' if tools_enabled else 'No'}\n",
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
                    temperature=temperature,
                    tools=TOOLS_CONFIG if tools_enabled else None
                ):
                    # 记录第一个token的时间
                    if first_token_time is None and (chunk_type == 'content' or chunk_type == 'reasoning'):
                        first_token_time = time.time()
                        first_token_latency = first_token_time - start_time

                    elapsed = time.time() - start_time
                    token_speed = token_count / elapsed if elapsed > 0 else 0
                    
                    # 计算延迟信息
                    latency_info = ""
                    if first_token_latency is not None:
                        latency_info = f"[dim]First token latency: {first_token_latency:.2f}s[/dim]\n"
                    
                    if chunk_type == 'function_call':
                        try:
                            # 解析函数调用信息
                            function_call = chunk
                            function_name = function_call.get('name')
                            arguments = function_call.get('arguments', '{}')
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                            
                            # 重命名参数 city 为 location
                            if 'city' in arguments:
                                arguments['location'] = arguments.pop('city')
                            
                            # 显示函数调用信息
                            console.print(Panel(
                                f"🔧 Calling function: [bold cyan]{function_name}[/bold cyan]\n" +
                                f"Arguments: [yellow]{json.dumps(arguments, indent=2)}[/yellow]",
                                title="Function Call",
                                border_style="cyan"
                            ))
                            
                            # 执行函数调用
                            if function_name == "get_weather":
                                result = get_weather(**arguments)
                                
                                # 显示函数返回结果
                                console.print(Panel(
                                    f"📊 Function returned:\n[green]{json.dumps(result, indent=2)}[/green]",
                                    title="Function Result",
                                    border_style="green"
                                ))
                                
                                # 将函数调用结果添加到消息历史
                                messages.append({
                                    "role": "function",
                                    "name": function_name,
                                    "content": json.dumps(result)
                                })
                        except Exception as e:
                            console.print(f"[red]Error executing function: {str(e)}[/red]")
                        continue
                    
                    # 原有的 reasoning 和 content 处理逻辑
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
