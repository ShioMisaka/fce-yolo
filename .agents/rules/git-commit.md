# Git Commit Conventions

所有提交使用中文，遵循 **Conventional Commits** 格式。

## 提交类型

| 类型 | 说明 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | 修复 bug |
| `docs:` | 文档变更 |
| `refactor:` | 代码重构（无功能变化） |
| `test:` | 添加或更新测试 |
| `chore:` | 维护任务、依赖、配置 |

## 规则

1. **标题行**：不超过 50 字符
2. **正文**：多文件变更时必须用列表列出
3. **无 AI 标识**：不包含 "生成于 AI"、"AI 协作" 等信息

## 示例

```bash
# 新功能
feat: 添加 BiCoordCrossAtt 双向坐标交叉注意力模块

# 修复 bug
fix: 修复 DFL 损失计算时的数值溢出问题

# 文档更新
docs: 更新 README.md 添加模块使用说明

# 代码重构
refactor: 重构测试文件到 script 目录
```
