from rich.console import Console
from rich.table import Column, Table

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Epoch")
table.add_column("Loss")
table.add_column("Time elapsed", justify="right")
table.add_column("Learning rate", justify="right")

data = [["1/20", "1.2454", "12.51s", "0.001"], 
        ["2/20", "1.2454", "12.51s", "0.001"]]

for row in data:
    table.add_row(*row)

console.print(table)