// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';

// Project GitHub Pages: https://muthuishere.github.io/browser-llm-nexus
// https://astro.build/config
export default defineConfig({
	site: 'https://muthuishere.github.io',
	base: '/browser-llm-nexus',
	integrations: [
		starlight({
			title: 'browser-llm-nexus',
			// Remove the right-hand "On this page" table of contents site-wide.
			tableOfContents: false,
			description:
				'Private LLM in any browser. GPU optional. Tool calling, embeddings, RAG and offline knowledge bundles over Transformers.js — no server, no bundled weights.',
			plugins: [
				starlightLlmsTxt({
					projectName: 'browser-llm-nexus',
					description:
						'Run an LLM in the browser — WebGPU when available, CPU otherwise. Tool calling, embeddings, RAG, and offline bundles.',
					details:
						'Tool calling (NexusChat), embeddings and vector search (NexusEmbedder, MemoryIndex), grounded answers over documents (NexusKnowledge), and zip export/import of models, indexes and whole knowledge bases for air-gapped use. Transformers.js is injectable; model sources are always explicit.',
				}),
			],
			customCss: ['@fontsource-variable/inter', './src/styles/deemwar.css'],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/muthuishere/browser-llm-nexus',
				},
			],
			sidebar: [
				{
					label: 'Start here',
					items: [
						{ label: 'Quickstart', slug: 'quickstart' },
						{ label: 'Live demo', link: '/browser-llm-nexus/demo/', attrs: { target: '_blank' } },
					],
				},
				{
					label: 'The differentiators',
					items: [
						{ label: 'GPU or CPU — same API', slug: 'gpu-or-cpu' },
						{ label: 'Tool calling', slug: 'tool-calling' },
						{ label: 'Knowledge & offline bundles', slug: 'knowledge' },
					],
				},
				{
					label: 'Building blocks',
					items: [
						{ label: 'Embeddings & RAG', slug: 'embeddings-rag' },
						{ label: 'Where models come from', slug: 'model-sources' },
					],
				},
				{
					label: 'Reference',
					items: [{ label: 'Together with toolnexus', slug: 'toolnexus' }],
				},
			],
		}),
	],
});
