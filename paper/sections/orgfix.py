import pathlib
p = next(f for f in sorted(pathlib.Path(__file__).resolve().parent.glob("*.md"))
         if f.name.startswith("01-"))
t = p.read_text(encoding="utf-8")
old = "§11 训练后量化方法综述与校准实践；§12 全栈实现与 APXInf 生态的角色。"
new = "§11 训练后量化方法综述与校准实践；§12 全栈实现与 APXInf 生态的角色；附录 A 结合核心源码走读全部实现并解析 APXInf 框架。"
assert old in t, "org paragraph anchor not found"
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("org paragraph updated")
