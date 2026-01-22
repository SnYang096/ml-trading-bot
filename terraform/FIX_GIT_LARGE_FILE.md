# 修复 Git 大文件问题

## 问题

手动下载的 Terraform provider 二进制文件（287 MB）被提交到 Git，超过了 GitHub 的 100 MB 限制。

## ✅ 已完成的步骤

1. ✅ 已将文件从 Git 索引中移除
2. ✅ 已更新 `.gitignore` 忽略 provider 文件

## ⚠️ 需要处理的问题

文件仍在 Git 历史记录中，需要从历史中完全移除才能推送到 GitHub。

## 🔧 解决方案

### 方案 1：使用 git filter-branch（推荐，如果只有你一个人在使用这个仓库）

```bash
# 1. 备份当前分支
git branch backup-before-cleanup

# 2. 从所有历史记录中移除大文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch -r terraform/terraform-provider-tencentcloud_1.82.58_linux_amd64/" \
  --prune-empty --tag-name-filter cat -- --all

# 3. 清理引用
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# 4. 强制垃圾回收
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. 强制推送到远程（⚠️ 会重写历史）
git push origin --force --all
git push origin --force --tags
```

### 方案 2：使用 git filter-repo（更现代的工具，需要先安装）

```bash
# 安装 git-filter-repo（如果未安装）
pip install git-filter-repo

# 从历史中移除文件
git filter-repo --path terraform/terraform-provider-tencentcloud_1.82.58_linux_amd64/ --invert-paths

# 强制推送
git push origin --force --all
```

### 方案 3：如果文件只在最近的提交中（最简单）

如果大文件只在最近的几个提交中，可以：

```bash
# 1. 重置到添加大文件之前的提交
git log --oneline --all | grep -B5 "terraform-provider"

# 2. 找到添加文件之前的提交哈希，然后：
git reset --soft <之前的提交哈希>

# 3. 重新提交（不包含大文件）
git commit -m "feat: terraform and et exp pass (removed large provider binary)"

# 4. 强制推送
git push origin --force
```

## 📝 当前状态

文件已从索引中移除，可以提交这个更改：

```bash
git add .gitignore
git commit -m "chore: remove large terraform provider binary from git and update .gitignore"
```

但**仍然无法推送到 GitHub**，因为历史记录中仍有大文件。

## 🎯 推荐操作步骤

1. **先提交当前的更改**（移除文件）：
   ```bash
   git add .gitignore
   git commit -m "chore: remove large terraform provider binary from git and update .gitignore"
   ```

2. **选择清理方案**：
   - 如果只有你一个人使用仓库：使用方案 1 或 2
   - 如果有其他人使用：需要协调，或者考虑创建新仓库

3. **清理后强制推送**（⚠️ 会重写历史）

## ⚠️ 重要提醒

- **强制推送会重写 Git 历史**，如果有其他人在使用这个仓库，需要通知他们
- 建议先备份仓库：`git clone --mirror <repo-url> backup-repo.git`
- 清理后，其他协作者需要重新克隆仓库或执行 `git fetch --all && git reset --hard origin/main`

## 📚 参考

- [GitHub: 处理大文件](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-a-repositorys-history)
- [git filter-branch 文档](https://git-scm.com/docs/git-filter-branch)
- [git-filter-repo 文档](https://github.com/newren/git-filter-repo)
