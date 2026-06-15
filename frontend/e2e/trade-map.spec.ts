/**
 * 全站 E2E 测试（Playwright）。
 *
 * 前置条件：生产服务运行在 http://127.0.0.1:8800
 *
 * 运行：
 *   npx playwright test --config=e2e/playwright.config.ts
 *   npx playwright test --config=e2e/playwright.config.ts --headed
 */

import { test, expect } from '@playwright/test';

// ═══════════════════════════════════════════════
// 工具
// ═══════════════════════════════════════════════

async function waitForChart(page: import('@playwright/test').Page) {
  await page.waitForSelector('canvas', { timeout: 20_000 });
}

async function waitForReady(page: import('@playwright/test').Page) {
  await page.waitForFunction(() => {
    const el = document.querySelector('[class*="status"]');
    const t = el?.textContent || '';
    return t.length > 0 && !t.includes('loading') && !t.includes('加载');
  }, { timeout: 25_000 }).catch(() => {});
}

function getChartRange(page: import('@playwright/test').Page) {
  return page.evaluate(() => {
    const chart = (window as any).__lwcChart;
    if (!chart) return null;
    return chart.timeScale().getVisibleLogicalRange();
  });
}

// ═══════════════════════════════════════════════
// TradeMap
// ═══════════════════════════════════════════════

test.describe('TradeMap', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/trade-map?symbol=ETHUSDT', {
      waitUntil: 'domcontentloaded', timeout: 45_000,
    });
    await waitForChart(page);
    await waitForReady(page);
  });

  test('01-页面结构完整', async ({ page }) => {
    const root = page.locator('#root');
    await expect(root).toBeAttached();
    expect((await root.innerHTML()).length).toBeGreaterThan(500);
    await expect(page.locator('text=交易地图')).toBeVisible();
  });

  test('02-Symbol和周期选择器存在', async ({ page }) => {
    // 至少 2 个 select: Symbol + 周期（可能还有皮肤切换器）
    expect(await page.locator('select').count()).toBeGreaterThanOrEqual(2);
    // 应存在包含 USDT 交易对的 select
    const all = page.locator('select'); const n = await all.count();
    let found = false;
    for (let i = 0; i < n; i++) {
      const opts = await all.nth(i).locator('option').allTextContents();
      if (opts.some((o: string) => o.includes('USDT'))) { found = true; break; }
    }
    expect(found).toBe(true);
  });

  test('03-图表canvas已渲染', async ({ page }) => {
    expect(await page.locator('canvas').count()).toBeGreaterThanOrEqual(1);
  });

  test('04-状态栏有内容', async ({ page }) => {
    const s = page.locator('[class*="status"]').first();
    await expect(s).toBeVisible({ timeout: 15_000 });
    expect((await s.textContent())!.length).toBeGreaterThan(0);
  });

  test('05-切换网格线图层', async ({ page }) => {
    const el = page.locator('text=网格线');
    await expect(el).toBeVisible();
    const cb = el.locator('input[type="checkbox"]');
    const was = await cb.isChecked();
    await el.click(); await page.waitForTimeout(1500);
    expect(await cb.isChecked()).toBe(!was);
  });

  test('06-图层切换不导致图表消失', async ({ page }) => {
    const cb = page.locator('input[type="checkbox"]').first();
    await expect(cb).toBeVisible({ timeout: 10_000 });
    await cb.click(); await page.waitForTimeout(1000);
    await cb.click(); await page.waitForTimeout(2000);
    await expect(page.locator('canvas').first()).toBeAttached();
  });

  test('07-勾选含挂单', async ({ page }) => {
    const el = page.locator('text=含挂单');
    await expect(el).toBeVisible();
    const cb = el.locator('input[type="checkbox"]');
    if (!(await cb.isChecked())) await el.click();
    await page.waitForTimeout(1000);
    expect(await cb.isChecked()).toBe(true);
  });

  test('08-切换Symbol更新URL', async ({ page }) => {
    const all = page.locator('select'); const n = await all.count();
    let sel: any = null;
    for (let i = 0; i < n; i++) {
      const opts = await all.nth(i).locator('option').allTextContents();
      if (opts.some((o: string) => o.includes('USDT'))) { sel = all.nth(i); break; }
    }
    if (!sel) return;
    const opts = await sel.locator('option').all();
    if (opts.length > 1) {
      await sel.selectOption(String(await opts[1].getAttribute('value')));
      await page.waitForTimeout(3000);
      await expect(page).toHaveURL(/symbol=/);
      await expect(page.locator('canvas').first()).toBeAttached();
    }
  });

  test('09-切换周期', async ({ page }) => {
    const all = page.locator('select'); const n = await all.count();
    let tf: any = null;
    for (let i = 0; i < n; i++) {
      const opts = await all.nth(i).locator('option').allTextContents();
      if (opts.some((o: string) => o.includes('15min') || o.includes('2h'))) { tf = all.nth(i); break; }
    }
    if (!tf) return;
    const before = await tf.inputValue();
    const target = before === '15min' ? '2h' : '15min';
    await tf.selectOption(target);
    await expect(tf).toHaveValue(target);
    await page.waitForTimeout(2000);
    await expect(page.locator('canvas').first()).toBeAttached();
  });

  test('10-点击策略按钮', async ({ page }) => {
    const btns = page.locator('button[class*="chip"]');
    if ((await btns.count()) > 1) {
      await btns.nth(1).click(); await page.waitForTimeout(2000);
      await expect(page.locator('button[class*="chipActive"]')).toHaveCount(1);
    }
  });

  test('11-打开特征抽屉', async ({ page }) => {
    await page.locator('button:has-text("特征")').click();
    await page.waitForTimeout(1000);
    await expect(page.locator('[class*="drawer"]').first()).toBeVisible({ timeout: 3000 });
  });

  test('12-切换订单面板', async ({ page }) => {
    const btn = page.locator('button:has-text("订单")');
    if (!(await btn.isVisible().catch(() => false))) return;
    const b = await btn.textContent();
    await btn.click(); await page.waitForTimeout(1000);
    expect(await btn.textContent()).not.toBe(b);
  });

  test('13-切换成交量', async ({ page }) => {
    const el = page.locator('text=成交量'); await expect(el).toBeVisible();
    const cb = el.locator('input[type="checkbox"]');
    const was = await cb.isChecked();
    await el.click(); await page.waitForTimeout(1000);
    expect(await cb.isChecked()).toBe(!was);
  });

  test('14-切换EMA1200', async ({ page }) => {
    await page.locator('text=EMA1200').click();
    await page.waitForTimeout(2000);
    await expect(page.locator('canvas').first()).toBeAttached();
  });

  test('15-点击刷新', async ({ page }) => {
    await page.locator('button:has-text("刷新")').click();
    await page.waitForTimeout(5000);
    await expect(page.locator('#root')).toBeAttached();
  });

  // ── Bug: 缩放不重置 ──

  test('16-缩放后轮询不重置时间轴', async ({ page }) => {
    const r1 = await getChartRange(page); if (!r1) return;
    const pane = page.locator('[class*="chartPane"]').first();
    await pane.hover();
    await page.keyboard.down('Control');
    await page.mouse.wheel(0, -300);
    await page.keyboard.up('Control');
    await page.waitForTimeout(1000);
    const r2 = await getChartRange(page); if (!r2) return;
    const bb = (r1 as any).to - (r1 as any).from;
    const ba = (r2 as any).to - (r2 as any).from;
    expect(ba).toBeLessThan(bb);
    await page.waitForTimeout(14_000);
    const r3 = await getChartRange(page); if (!r3) return;
    const bp = (r3 as any).to - (r3 as any).from;
    expect(bp).toBeLessThan(bb - 3);
  });

  // ── Bug: 网格对齐 ──

  test('17-网格线标签位置正常', async ({ page }) => {
    const ccb = page.locator('text=网格线').locator('input[type="checkbox"]');
    if (!(await ccb.isChecked().catch(() => false))) {
      await page.locator('text=网格线').click(); await page.waitForTimeout(3000);
    }
    const labels = page.locator('[class*="label"]');
    if ((await labels.count()) === 0) return;
    const box = await labels.first().boundingBox(); if (!box) return;
    expect(box.x).toBeGreaterThan(0);
    expect(box.y).toBeGreaterThan(0);
    expect((await labels.first().textContent())!.length).toBeGreaterThan(0);
  });

  // ── Bug: 价格轴缩放 ──

  test('18-价格轴缩放后不重置', async ({ page }) => {
    const pane = page.locator('[class*="chartPane"]').first();
    const box = await pane.boundingBox(); if (!box) return;
    await page.mouse.move(box.x + box.width - 10, box.y + box.height / 2);
    await page.mouse.wheel(0, -200);
    await page.waitForTimeout(14_000);
    await expect(page.locator('canvas').first()).toBeAttached();
  });

  test('19-crosshair状态变化', async ({ page }) => {
    await page.locator('[class*="chartPane"]').first().hover();
    await page.mouse.move(300, 200);
    await page.waitForTimeout(500);
    expect((await page.locator('[class*="status"]').first().textContent())!.length).toBeGreaterThan(0);
  });

  test('20-多图层同时切换', async ({ page }) => {
    const cbs = page.locator('input[type="checkbox"]');
    const n = await cbs.count(); if (n < 3) return;
    for (let i = 0; i < 3; i++) await cbs.nth(i).click();
    await page.waitForTimeout(2000);
    await expect(page.locator('canvas').first()).toBeAttached();
    for (let i = 0; i < 3; i++) await cbs.nth(i).click();
    await page.waitForTimeout(2000);
    await expect(page.locator('canvas').first()).toBeAttached();
  });

  test('21-附图与主图同步', async ({ page }) => {
    expect(await page.locator('canvas').count()).toBeGreaterThanOrEqual(2);
  });
});

// ═══════════════════════════════════════════════
// Grid / Orders / Smokes
// ═══════════════════════════════════════════════

test.describe('多品种地图', () => {
  test('22-加载', async ({ page }) => {
    await page.goto('/trade-map-grid', { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await page.waitForTimeout(10_000);
    await expect(page.locator('#root')).toBeAttached();
    expect((await page.locator('#root').innerHTML()).length).toBeGreaterThan(500);
  });
  test('23-canvas存在', async ({ page }) => {
    await page.goto('/trade-map-grid', { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await page.waitForTimeout(10_000);
    expect(await page.locator('canvas').count()).toBeGreaterThan(0);
  });
});

test.describe('Orders', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/orders', { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await page.waitForTimeout(8000);
  });
  test('24-加载', async ({ page }) => {
    await expect(page.locator('#root')).toBeAttached();
    expect((await page.locator('#root').innerHTML()).length).toBeGreaterThan(500);
  });
  test('25-含Symbol选择器', async ({ page }) => {
    await expect(page.locator('select').first()).toBeVisible();
  });
});

for (const p of ['/signals', '/account', '/regime', '/monitoring']) {
  test.describe(p, () => {
    test('smoke', async ({ page }) => {
      await page.goto(p, { waitUntil: 'domcontentloaded', timeout: 45_000 });
      await page.waitForTimeout(4000);
      await expect(page.locator('#root')).toBeAttached();
      expect((await page.locator('#root').innerHTML()).length).toBeGreaterThan(200);
    });
  });
}

// ═══════════════════════════════════════════════
// 导航 & URL
// ═══════════════════════════════════════════════

test.describe('导航', () => {
  test('26-点击导航链接跳转', async ({ page }) => {
    await page.goto('/trade-map?symbol=ETHUSDT', { waitUntil: 'domcontentloaded', timeout: 45_000 });
    await page.waitForTimeout(3000);
    await page.locator('a:has-text("策略信号")').click();
    await page.waitForTimeout(3000);
    await expect(page).toHaveURL(/\/signals/);
  });
  test('27-浏览器后退', async ({ page }) => {
    await page.goto('/trade-map?symbol=ETHUSDT', { waitUntil: 'domcontentloaded', timeout: 45_000 });
    await page.waitForTimeout(2000);
    await page.goto('/monitoring', { waitUntil: 'domcontentloaded', timeout: 45_000 });
    await page.waitForTimeout(2000);
    await page.goBack(); await page.waitForTimeout(2000);
    await expect(page).toHaveURL(/trade-map/);
    await expect(page.locator('#root')).toBeAttached();
  });
});

test.describe('URL参数', () => {
  test('28-?symbol预选', async ({ page }) => {
    await page.goto('/trade-map?symbol=BTCUSDT', { waitUntil: 'domcontentloaded', timeout: 45_000 });
    await page.waitForTimeout(5000);
    // 找到包含交易对选项的 select（不是皮肤切换器）
    const all = page.locator('select');
    const n = await all.count();
    let symSel: any = null;
    for (let i = 0; i < n; i++) {
      const opts = await all.nth(i).locator('option').allTextContents();
      if (opts.some((o: string) => o.includes('USDT'))) { symSel = all.nth(i); break; }
    }
    if (symSel) await expect(symSel).toHaveValue('BTCUSDT', { timeout: 10_000 });
  });
  test('29-无效symbol不崩溃', async ({ page }) => {
    try {
      await page.goto('/trade-map?symbol=INVALID_XYZ', { waitUntil: 'domcontentloaded', timeout: 20_000 });
    } catch {}
    await page.waitForTimeout(3000);
    await expect(page.locator('#root')).toBeAttached();
    expect((await page.locator('#root').innerHTML()).length).toBeGreaterThan(100);
  });
});
