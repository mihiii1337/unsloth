// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const voiceTab = readFileSync(
  new URL("../src/features/settings/tabs/voice-tab.tsx", import.meta.url),
  "utf8",
);
const adapter = readFileSync(
  new URL(
    "../src/features/chat/adapters/studio-model-dictation-adapter.ts",
    import.meta.url,
  ),
  "utf8",
);

test("changing the dictation device releases only this tab's model", () => {
  // Unscoped, this races another surface: the unload lands after that surface
  // swapped the resident model and tears down one this tab never owned.
  assert.match(
    voiceTab,
    /void unloadSttModel\(sttEngineFor\(sttModel\), sttModel\)\.catch\(/,
  );
});

test("the unload API still takes the engine and model that scoping needs", () => {
  assert.match(
    adapter,
    /export function unloadSttModel\(\s*engine\?: SttEngine,\s*model\?: string,\s*\)/,
  );
  assert.match(adapter, /if \(model\) params\.set\("model", model\)/);
});

test("the device preference travels with every load and transcribe", () => {
  // A load that omits it is read as "no opinion" server-side, so the setting
  // would silently never apply.
  assert.match(adapter, /device: resolvedDevice/);
  assert.match(adapter, /params\.set\("device", settings\.sttDevice\)/);
});
