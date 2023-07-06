import { defineConfig } from "tinacms";

// Your hosting provider likely exposes this as an environment variable
const branch = process.env.HEAD || process.env.VERCEL_GIT_COMMIT_REF || "main";

export default defineConfig({
  branch,
  clientId: "547f2274-594f-4d58-ab6c-883715fd0058", // Get this from tina.io
  token: "1e01e096ad60852c704ff787065cafbe7439a59a", // Get this from tina.io

  build: {
    outputFolder: "admin",
    publicFolder: "static",
  },
  media: {
    tina: {
      mediaRoot: "",
      publicFolder: "assets",
    },
  },
  schema: {
    collections: [
      {
        name: "blog",
        label: "Blog",
        path: "content/blog",
        fields: [
          {
            type: "string",
            name: "title",
            label: "Title",
            isTitle: true,
            required: true,
          },
           {
            type: "datetime",
            label: "Date",
            name: "date",
          },
          {
            type: "boolean",
            name: "draft",
            label: "Draft",
          },
          {
            type: "string",
            name: "description",
            label: "Description",
          },
          {
            type: "string",
            name: "slug",
            label: "Slug",
            required: true,
          },
          {
            type: "string",
            name: "url",
            label: "URL in Format (blog/yyyy/mm/dd/slug/)",
            required: true,
          },
          {
            type: "string",
            name: "type",
            label: "Type",
          },
          {
            label: 'Categories',
            name: 'Categories',
            list: true,
            defaultItem: 'Programming',
            type: 'string'
          },
          {
            label: 'Keywords',
            name: 'Keywords',
            component: 'list',
            defaultItem: 'python',
            list: true,
            type: 'string'
            
          },
          {
            type: "string",
            name: "thumbnail",
            label: "thumbnail",
            required: true,
          },
          {
            type: "string",
            name: "image",
            label: "image",
            required: true,
          },
          {
            type: "rich-text",
            name: "body",
            label: "Body",
            isBody: true,
          },
        ],
      },
    ],
  },
});
