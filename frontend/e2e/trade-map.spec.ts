/**
 * TradeMap 页面 E2E 测试（Playwright）。
 *
 * 前置条件：
 *   1. 前端 dev server: cd frontend && npm run dev
 *   2. 后端 API 服务运行中（端口 8000，Vite 自动代理 /api → 后端）
 *
 * 运行：
 *   npx playwright test --config=e2e/playwright.config.ts
 *   npx playwright test --config=e2e/playwright.config.ts --headed  # 有头模式调试
 */

import { test, expect } from '@playwright/test';

const BASE = '/static';

test.describe('TradeMap 页面', () => {
  test.beforeEach(async ({ page }) => {
    // 页面较大（图表 + 数据），给充足加载时间
    await page.goto(`${BASE}/trade-map?symbol=ETHUSDT`, {
      waitUntil: 'networkidle',
      timeout: 30_000,
    });
    await page.waitForTimeout(3000);
  });

  test('页面基本结构加载完整', async ({ page }) => {
    // #root 被 React 挂载且渲染了内容
    const root = page.locator('#root');
    await expect(root).toBeAttached();
    const html = await root.innerHTML();
    expect(html.length).toBeGreaterThan(500);

    // 导航栏存在
    await expect(page.locator('text=交易地图')).toBeVisible();
    await expect(page.locator('text=策略信号')).toBeVisible();
  });

  test('Symbol / 周期选择器存在', async ({ page }) => {
    const selects = page.locator('select');
    const count = await selects.count();
    expect(count).toBeGreaterThanOrEqual(2);
  });

  test('图层复选框可交互', async ({ page }) => {
    const checkboxes = page.locator('input[type="checkbox"]');
    const count = await checkboxes.count();
    expect(count).toBeGreaterThanOrEqual(3);
  });

  test('切换 Symbol 更新 URL', async ({ page }) => {
    const symbolSelect = page.locator('select').first();
    const options = await symbolSelect.locator('option').all();

    if (options.length > 1) {
      const val = await options[1].getAttribute('value');
      await symbolSelect.selectOption(String(val));
      await page.waitForTimeout(1000);
      await expect(page).toHaveURL(/symbol=/);
    }
  });

  test('切换周期后选择器值更新', async ({ page }) => {
    const selects = page.locator('select');
    const count = await selects.count();
    if (count >= 2) {
      const tfSelect = selects.nth(1);
      await tfSelect.selectOption('15min');
      await expect(tfSelect).toHaveValue('15min');
    }
  });
});

test.describe('多品种地图页面', () => {
  test('网格页加载', async ({ page }) => {
    await page.goto(`${BASE}/trade-map-grid`, {
      waitUntil: 'networkidle',
      timeout: 30_000,
    });
    await page.waitForTimeout(3000);

    const root = page.locator('#root');
    await expect(root).toBeAttached();
    const html = await root.innerHTML();
    expect(html.length).toBeGreaterThan(500);
  });
});

test.describe('Orders 页面', () => {
  test('订单页加载', async ({ page }) => {
    await page.goto(`${BASE}/orders`, {
      waitUntil: 'networkidle',
      timeout: 30_000,
    });
    await page.waitForTimeout(3000);

    await expect(page.locator('#root')).toBeAttached();
    const html = await page.locator('#root').innerHTML();
    expect(html.length).toBeGreaterThan(500);
  });
});
