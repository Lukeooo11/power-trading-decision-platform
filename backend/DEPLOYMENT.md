# 云端后端部署

## Render

在仓库根目录创建 Render Web Service，Runtime 选择 Docker，Dockerfile 使用 `backend/Dockerfile`，端口由 `PORT` 环境变量注入。

环境变量：

```text
POWER_TRADING_ENV=production
POWER_TRADING_CORS_ORIGINS=https://lukeooo11.github.io
```

健康检查：`/api/v1/health`

部署完成后，用浏览器访问 `https://<service>.onrender.com/api/v1/health`，确认返回 JSON 后，再把前端构建前的 `window.__PLATFORM_API_BASE__` 设置为该地址。

## 数据边界

生产环境不得提交客户原始文件、真实名称映射、SQLite 数据库、API 密钥或日志。当前 Dockerfile 仅复制脱敏 `private-data` 和后端代码；正式客户环境应改为对象存储或数据库注入。

## 长任务

价格预测接口创建运行后可能需要数分钟。前端使用 `run_id` 轮询状态；云平台需要将请求超时设置为至少 300 秒，或后续将模型计算迁移到独立 Worker。
