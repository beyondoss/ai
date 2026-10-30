import { defineTool } from "eve/tools";
import { z } from "zod";

// endsTurn: true, but it throws: the model sees the error and the run continues.
export default defineTool({
  description: "Always fails.",
  inputSchema: z.object({}),
  endsTurn: true,
  async execute() {
    throw new Error("boom-failure");
  },
});
