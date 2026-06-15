import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E 测试配置。
 *
 * 前置条件：
 *   1. 前端 dev server 已启动: cd frontend && npm run dev
 *   2. 安装 Playwright:        npx playwright install chromium
 *
 * 运行：
 *   npx playwright test
 *   npx playwright test --ui      (交互模式)
 *   npx playwright test --headed  (有头模式，方便调试)
 *
 * CI 模式：
 *   npx playwright test --reporter=html
 */

const PORT = 5173;
const BASE_URL = `http://localhost:${PORT}/static`;

export default defineConfig({
  testDir: './',
  timeout: 90_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [['html', { open: 'never' }], ['list']],

  use: {
    baseURL: BASE_URL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  // dev server 由外部启动，此处不自动启停
  webServer: undefined,
});
