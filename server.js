const Hapi = require('@hapi/hapi');
const Boom = require('@hapi/boom');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');
const admin = require('firebase-admin');

// Initialize Firebase Admin SDK
const serviceAccount = require('./submissionmlgc-rifkimaulana-firebase-adminsdk-7czo4-dc02f38cd0.json'); // Replace with the path to your Firebase service account key
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    databaseURL: 'https://submissionmlgc-rifkimaulana.firebaseio.com' // Replace with your Firebase project URL
});

// Get Firestore database reference
const db = admin.firestore();

const init = async () => {
    const server = Hapi.server({
        port: 3000,
        host: 'localhost',
        routes: {
            cors: {
                origin: ["Access-Control-Allow-Origin", "http://localhost:8080"],
                headers: ["Accept", "Content-Type"],
                additionalHeaders: ["X-Requested-With"]
            }
        }
    });

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
            const {image} = request.payload;
            if (!image || !image._data) throw Boom.badRequest('File is required');

            try {
                const tensor = tf.node.decodeImage(image._data, 3)
                    .resizeBilinear([224, 224])
                    .expandDims(0)
                    .div(255.0);

                const prediction = model.predict(tensor).arraySync();
                const classification = prediction[0] > 0.58 ? 'Cancer' : 'Non-cancer';

                // Create a new prediction document in Firestore
                const predictionId = uuidv4();
                const predictionDoc = {
                    id: predictionId,
                    result: classification,
                    suggestion: classification === 'Cancer'
                        ? 'Segera periksa ke dokter!'
                        : 'Penyakit kanker tidak terdeteksi.',
                    createdAt: new Date().toISOString(),
                };

                // Save the prediction to Firestore
                await db.collection('predictions').doc(predictionId).set(predictionDoc);

                return h.response({
                    status: 'success',
                    message: 'Model is predicted successfully',
                    data: predictionDoc,
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
            return h.response({status: 'fail', message}).code(statusCode);
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
