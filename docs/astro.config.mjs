// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/simgeometry/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'simgeometry',
	title: 'simgeometry',
	description: 'Shapes, structured grids, and domains for numerical simulations.',
	pythonPackage: 'simgeometry',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, from the shapes to the grids, and a Numerics section on each method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'simgeometry.shapes': 'Shape conventions',
			'simgeometry.objects.line': 'Line',
			'simgeometry.objects.circle': 'Circle',
			'simgeometry.objects.sphere': 'Sphere',
			'simgeometry.objects.rectangle': 'Rectangle',
			'simgeometry.objects.cuboid': 'Cuboid',
			'simgeometry.objects.polygon': 'Polygon',
			'simgeometry.objects.triangle': 'Triangle',
			'simgeometry.objects.aperture2d': 'Aperture2D',
			'simgeometry.objects.aperture3d': 'Aperture3D',
			'simgeometry.objects.axisymmetric': 'Solids of revolution',
			'simgeometry.objects.cylinder': 'Cylinder',
			'simgeometry.objects.tube': 'Tube',
			'simgeometry.objects.cone': 'Cone',
			'simgeometry.objects.torus': 'Torus',
			'simgeometry.objects.slanted_tube': 'SlantedTube',
			'simgeometry.objects.triangle_mesh': 'TriangleMesh',
			'simgeometry.objects.fourier_surface': 'FourierSurface',
			'simgeometry.operations': 'Booleans and transforms',
			'simgeometry.collection': 'Collections',
			'simgeometry.panels': 'Boundary panels',
			'simgeometry.crossings': 'First crossings',
			'simgeometry.kernels.polygon_hits': 'Polygon hit test',
			'simgeometry.kernels.triangle_hits': 'Triangle hit test',
			'simgeometry.polygon_objects': 'Swept absorption',
			'simgeometry.queries': 'Edge crossings',
			'simgeometry.cut_edges': 'Cut edges',
			'simgeometry.fractions': 'Cut-cell fractions',
			'simgeometry.grid': 'Grid',
			'simgeometry.domain_decomposition': 'Domain decomposition',
			'simgeometry.simulation_domain': 'Particle boundaries',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/simgeometry/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
				{ label: 'Geometry inspection and queries', slug: 'getting-started/geometry-api' },
			],
		},
		{
			label: 'Examples',
			items: [{ autogenerate: { directory: 'examples' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
