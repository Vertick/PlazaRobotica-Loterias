from pathlib import Path
import json

def scan_project(root: Path):
    files = []
    tree = {}

    def walk(dir_path: Path, tree_node: dict):
        tree_node["__files__"] = []
        tree_node["__dirs__"] = {}

        for item in sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if item.is_file():
                files.append({
                    "absolute_path": str(item.resolve()),
                    "relative_path": str(item.relative_to(root)),
                    "name": item.name,
                    "size_bytes": item.stat().st_size
                })
                tree_node["__files__"].append(item.name)

            elif item.is_dir():
                tree_node["__dirs__"][item.name] = {}
                walk(item, tree_node["__dirs__"][item.name])

    walk(root, tree)
    return files, tree


def pretty_print_tree(tree: dict, indent: str = "") -> str:
    lines = []

    for dirname, subtree in tree.get("__dirs__", {}).items():
        lines.append(f"{indent}{dirname}/")
        lines.extend(pretty_print_tree(subtree, indent + "  ").splitlines())

    for filename in tree.get("__files__", []):
        lines.append(f"{indent}{filename}")

    return "\n".join(lines)


if __name__ == "__main__":
    ROOT = Path(".").resolve()

    files, tree = scan_project(ROOT)

    # 1?? Guardar listado de archivos
    with open("project_files.json", "w", encoding="utf-8") as f:
        json.dump(files, f, indent=2, ensure_ascii=False)

    # 2?? Guardar árbol de carpetas (texto)
    tree_text = pretty_print_tree(tree)
    with open("project_tree.txt", "w", encoding="utf-8") as f:
        f.write(tree_text)

    print("? Escaneo completado")
    print(f"  - Archivos encontrados: {len(files)}")
    print("  - project_files.json")
    print("  - project_tree.txt")