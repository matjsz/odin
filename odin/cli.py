import click


@click.command()
@click.argument("name")
def main(name):
    """A simple CLI tool."""
    print(f"Hello, {name}!")


if __name__ == "__main__":
    main()
