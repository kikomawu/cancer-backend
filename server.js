const Hapi = require('@hapi/hapi');
const Boom = require('@hapi/boom');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');

const init = async () => {
    const server = Hapi.server({ port: 3000, host: 'localhost' });

    const modelUrl = 'https://storage.googleapis.com/rifki-bucket-model/model.json';
    let model;

    try {
        console.log('Loading TensorFlow.js model...');
        model = await tf.loadGraphModel(modelUrl);
        console.log('Model loaded successfully!');
    } catch (error) {
        console.error('Error loading model:', error);
        process.exit(1);
    }

    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                maxBytes: 1000000,
                output: 'stream',
                parse: true,
                multipart: true,
                allow: 'multipart/form-data',
            },
        },
        handler: async (request, h) => {
            const { file } = request.payload;
            if (!file || !file._data) throw Boom.badRequest('File is required');

            try {
                const tensor = tf.node.decodeImage(file._data, 3)
                    .resizeBilinear([224, 224])
                    .expandDims(0)
                    .div(255.0);

                const prediction = model.predict(tensor).arraySync();
                const classification = prediction[0] > 0.58 ? 'Cancer' : 'Non-cancer';

                return h.response({
                    status: 'success',
                    message: 'Model is predicted successfully',
                    data: {
                        id: uuidv4(),
                        result: classification,
                        suggestion: classification === 'Cancer'
                            ? 'Segera periksa ke dokter!'
                            : 'Penyakit kanker tidak terdeteksi.',
                        createdAt: new Date().toISOString(),
                    },
                }).code(200);
            } catch {
                throw Boom.badRequest('Terjadi kesalahan dalam melakukan prediksi');
            }
        },
    });

    server.ext('onPreResponse', (request, h) => {
        const response = request.response;
        if (response.isBoom) {
            const statusCode = response.output.statusCode;
            const message = statusCode === 413
                ? 'Payload content length greater than maximum allowed: 1000000'
                : 'Terjadi kesalahan dalam melakukan prediksi';
            return h.response({ status: 'fail', message }).code(statusCode);
        }
        return h.continue;
    });

    await server.start();
    console.log('Server running on %s', server.info.uri);
};

process.on('unhandledRejection', (err) => {
    console.error(err);
    process.exit(1);
});

init();
