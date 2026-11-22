/** The address space the bus reads. A software store, not a DIMM. */
export const RAM_BYTES = 64 * 1024;
/** Backing store. Pages move into RAM; nothing runs from the SSD. */
export const SSD_BYTES = 1024 * 1024;
export const SSD_PAGE = 4096;
/** Slower than the 32-byte bus, on purpose. */
export const SSD_BYTES_PER_CYCLE = 4;

export class Board {
  /** Host Pi statements retired. The kernel body is not the CPU. */
  cpu = 0;
  ssdBytes = 0;
  ssdCycles = 0;
  ssdPages = 0;
  programBytes = 0;

  /** Copy the program into RAM. Returns the first free address. */
  pageIn(bytes: number): number {
    if (bytes < 0 || bytes > SSD_BYTES) throw new Error("SSD: program does not fit");
    this.programBytes = bytes;
    this.ssdBytes = bytes;
    this.ssdPages = bytes === 0 ? 0 : Math.ceil(bytes / SSD_PAGE);
    this.ssdCycles = bytes === 0 ? 0 : Math.max(1, Math.ceil(bytes / SSD_BYTES_PER_CYCLE));
    const heap = bytes === 0 ? 0 : Math.ceil(bytes / 8) * 8;
    if (heap > RAM_BYTES) throw new Error("RAM: program does not fit");
    return heap;
  }

  retire() {
    this.cpu += 1;
  }
}
