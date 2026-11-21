// What tiny_server.mjs serves, shared with check.mjs (importing the server would start it).
export const LISTING_UI = {
  csp: {
    connectDomains: ["https://listing.example.test"],
    resourceDomains: ["https://cdn.listing.example.test"],
  },
  permissions: { camera: {}, microphone: {} },
  prefersBorder: true,
};
export const CONTENT_UI = {
  csp: { connectDomains: ["https://content.example.test"] },
};
