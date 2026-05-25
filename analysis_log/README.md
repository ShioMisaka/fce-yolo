# Analysis Log

本目录用于存放开发过程中的分析和修复日志。

## 目录结构

```
analysis_log/
├── README.md                           # 本文件
└── 2026-03-26-training-fix/            # 2026-03-26 训练脚本修复日志
    ├── README.md                       # 修复概述
    ├── TRAINING_ISSUES_ANALYSIS.md     # 问题分析报告
    └── FIX_SUMMARY.md                  # 修复总结
```

## 命名规范

使用日期命名格式：`YYYY-MM-DD-描述/`

例如：

- `2026-03-26-training-fix/` - 训练脚本修复
- `2026-03-27-module-analysis/` - 模块分析
- `2026-03-28-bug-investigation/` - Bug 调查

## Git 设置

本目录下的日期文件夹已添加到 `.gitignore`：

```gitignore
analysis_log/20??-??-??-*/
```

所有以日期格式命名的文件夹都不会被提交到 git 仓库，仅用于本地开发记录。

## 创建新日志

1. 在 `analysis_log/` 下创建新文件夹：

   ```bash
   mkdir analysis_log/2026-03-27-your-topic
   ```

2. 在文件夹中添加必要的文档：
   - `README.md` - 概述和快速参考
   - 其他分析文档

3. 文件夹会自动被 git 忽略，无需手动更新 `.gitignore`
