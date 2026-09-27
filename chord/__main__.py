"""Command-line scoring: ``python -m chord --generated gen.jsonl --reference ref.jsonl``."""

from .api import main

if __name__ == "__main__":
    main()
