import { defineTool } from "eve/tools";
import { z } from "zod";

// endsTurn: true — the run ends after this tool succeeds.
export default defineTool({
  description: "Finish the turn with a short closing note.",
  inputSchema: z.object({}),
  endsTurn: true,
  async execute() {
    return { done: true };
  },
});
