import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description='Processing args')
    parser.add_argument('--input', type=str, help='The input file')
    parser.add_argument('--output', type=str, help='The output file')
