/**
 * softwell-machine — scheduler: block-paged KV cache (worker-local).
 *
 * One instance lives per pipeline stage, owned by the engine's
 * StageExecutor. Physical K/V for a layer lives in fixed-size blocks;
 * each sequence has a block table per layer mapping logical block index
 * -> physical block id. Blocks are allocated lazily (a block's per-layer
 * K/V arrays are only created on first append to that layer) and
 * returned to a free-list when the sequence is freed.
 *
 * Key/value vectors are `numKvHeads * headDim` floats per token (GQA:
 * numKvHeads may be < numAttentionHeads).
 */

import type { PagedKVCache } from "../contracts.ts";

interface Block {
  /** layer -> keys for this block, shape [blockSize, kvDim], row-major. */
  keysByLayer: Map<number, Float32Array>;
  /** layer -> values for this block, same shape. */
  valuesByLayer: Map<number, Float32Array>;
}

export class BlockPagedKVCache implements PagedKVCache {
  readonly blockSize: number;
  private readonly total: number;
  private readonly numKvHeads: number;
  private readonly headDim: number;
  private readonly kvDim: number;
  private readonly blocks: Block[];
  /** Stack of free physical block ids. */
  private readonly freeList: number[];
  /**
   * seqId -> layer -> ordered list of physical block ids (the block table).
   * Table entry i covers logical positions [i * blockSize, (i+1) * blockSize).
   */
  private readonly tables: Map<string, Map<number, number[]>>;

  constructor(blockSize: number, totalBlocks: number, numKvHeads: number, headDim: number) {
    if (!Number.isInteger(blockSize) || blockSize <= 0) {
      throw new Error(`BlockPagedKVCache: blockSize must be a positive integer, got ${blockSize}`);
    }
    if (!Number.isInteger(totalBlocks) || totalBlocks <= 0) {
      throw new Error(`BlockPagedKVCache: totalBlocks must be a positive integer, got ${totalBlocks}`);
    }
    if (!Number.isInteger(numKvHeads) || numKvHeads <= 0) {
      throw new Error(`BlockPagedKVCache: numKvHeads must be a positive integer, got ${numKvHeads}`);
    }
    if (!Number.isInteger(headDim) || headDim <= 0) {
      throw new Error(`BlockPagedKVCache: headDim must be a positive integer, got ${headDim}`);
    }
    this.blockSize = blockSize;
    this.total = totalBlocks;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.kvDim = numKvHeads * headDim;
    this.blocks = [];
    this.freeList = [];
    for (let i = 0; i < totalBlocks; i++) {
      this.blocks.push({ keysByLayer: new Map(), valuesByLayer: new Map() });
      this.freeList.push(i);
    }
    this.tables = new Map();
  }

  /**
   * Grow the sequence's block table for `layer` so it can hold `numTokens`
   * tokens, allocating new blocks from the free-list as needed. Returns the
   * full block table for that layer (physical block ids, in order).
   * Throws on out-of-memory (no free blocks).
   */
  async allocate(seqId: string, layer: number, numTokens: number): Promise<number[]> {
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error(`BlockPagedKVCache.allocate: numTokens must be a non-negative integer, got ${numTokens}`);
    }
    let perLayer = this.tables.get(seqId);
    if (perLayer === undefined) {
      perLayer = new Map();
      this.tables.set(seqId, perLayer);
    }
    let table = perLayer.get(layer);
    if (table === undefined) {
      table = [];
      perLayer.set(layer, table);
    }
    const needed = Math.ceil(numTokens / this.blockSize);
    while (table.length < needed) {
      const blockId = this.freeList.pop();
      if (blockId === undefined) {
        throw new Error(
          `BlockPagedKVCache out of memory: no free blocks (total=${this.total}, used=${this.usedBlocks()}) ` +
            `while allocating for seq ${seqId} layer ${layer}`,
        );
      }
      table.push(blockId);
    }
    return [...table];
  }

  /** Write one token's K/V for `layer` at absolute `position`. */
  async append(seqId: string, layer: number, position: number, key: Float32Array, value: Float32Array): Promise<void> {
    if (!Number.isInteger(position) || position < 0) {
      throw new Error(`BlockPagedKVCache.append: position must be a non-negative integer, got ${position}`);
    }
    if (key.length !== this.kvDim || value.length !== this.kvDim) {
      throw new Error(
        `BlockPagedKVCache.append: key/value must have ${this.kvDim} floats ` +
          `(numKvHeads=${this.numKvHeads} * headDim=${this.headDim}), ` +
          `got key.length=${key.length}, value.length=${value.length}`,
      );
    }
    // Grow the table so the block for this position exists.
    const table = await this.allocate(seqId, layer, position + 1);
    const blockIndex = Math.floor(position / this.blockSize);
    const offsetInBlock = position % this.blockSize;
    const blockId = table[blockIndex] as number;
    const block = this.blocks[blockId] as Block;
    let keys = block.keysByLayer.get(layer);
    if (keys === undefined) {
      keys = new Float32Array(this.blockSize * this.kvDim);
      block.keysByLayer.set(layer, keys);
    }
    let values = block.valuesByLayer.get(layer);
    if (values === undefined) {
      values = new Float32Array(this.blockSize * this.kvDim);
      block.valuesByLayer.set(layer, values);
    }
    const start = offsetInBlock * this.kvDim;
    keys.set(key, start);
    values.set(value, start);
  }

  /**
   * Concatenated K and V for `positions` (in the requested order) for one
   * layer. Each returned array has shape [positions.length, kvDim].
   */
  async read(seqId: string, layer: number, positions: number[]): Promise<{ keys: Float32Array; values: Float32Array }> {
    const perLayer = this.tables.get(seqId);
    const table = perLayer?.get(layer);
    if (table === undefined) {
      throw new Error(`BlockPagedKVCache.read: no blocks allocated for seq ${seqId} layer ${layer}`);
    }
    const keys = new Float32Array(positions.length * this.kvDim);
    const values = new Float32Array(positions.length * this.kvDim);
    for (let i = 0; i < positions.length; i++) {
      const position = positions[i] as number;
      if (!Number.isInteger(position) || position < 0) {
        throw new Error(`BlockPagedKVCache.read: position must be a non-negative integer, got ${position}`);
      }
      const blockIndex = Math.floor(position / this.blockSize);
      const offsetInBlock = position % this.blockSize;
      const blockId = table[blockIndex];
      if (blockId === undefined) {
        throw new Error(
          `BlockPagedKVCache.read: position ${position} not allocated for seq ${seqId} layer ${layer}`,
        );
      }
      const block = this.blocks[blockId] as Block;
      const blockKeys = block.keysByLayer.get(layer);
      const blockValues = block.valuesByLayer.get(layer);
      if (blockKeys === undefined || blockValues === undefined) {
        throw new Error(
          `BlockPagedKVCache.read: no K/V written yet for seq ${seqId} layer ${layer} position ${position}`,
        );
      }
      const start = offsetInBlock * this.kvDim;
      keys.set(blockKeys.subarray(start, start + this.kvDim), i * this.kvDim);
      values.set(blockValues.subarray(start, start + this.kvDim), i * this.kvDim);
    }
    return { keys, values };
  }

  /** Release every block owned by `seqId`, across all layers. No-op if unknown. */
  async free(seqId: string): Promise<void> {
    const perLayer = this.tables.get(seqId);
    if (perLayer === undefined) return;
    for (const table of perLayer.values()) {
      for (const blockId of table) {
        const block = this.blocks[blockId] as Block;
        // Drop lazily-allocated per-layer arrays so a reused block starts clean.
        block.keysByLayer.clear();
        block.valuesByLayer.clear();
        this.freeList.push(blockId);
      }
    }
    this.tables.delete(seqId);
  }

  /** Release everything: all sequences, all blocks back on the free-list. */
  async close(): Promise<void> {
    for (const perLayer of this.tables.values()) {
      for (const table of perLayer.values()) {
        for (const blockId of table) {
          const block = this.blocks[blockId] as Block;
          block.keysByLayer.clear();
          block.valuesByLayer.clear();
          this.freeList.push(blockId);
        }
      }
    }
    this.tables.clear();
  }

  usedBlocks(): number {
    return this.total - this.freeList.length;
  }

  totalBlocks(): number {
    return this.total;
  }
}
