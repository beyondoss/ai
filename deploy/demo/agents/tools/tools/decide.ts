import { defineTool } from "eve/tools";
import { z } from "zod";

// endsTurn as a function of the validated output: the run ends only when stop is true.
export default defineTool({
  description: "Stop only when asked.",
  inputSchema: z.object({ stop: z.boolean() }),
  endsTurn: (output) => output.stop === true,
  async execute({ stop }) {
    return { stop };
  },
});
