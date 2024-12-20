const {SecretManagerServiceClient} = require('@google-cloud/secret-manager');
const admin = require('firebase-admin');
const {v4: uuidv4} = require('uuid');
const tf = require('@tensorflow/tfjs-node');
const Boom = require('@hapi/boom');
const Hapi = require('@hapi/hapi');

const client = new SecretManagerServiceClient();

async function getFirebaseCredentials() {
    const secretName = 'projects/submissionmlgc-rifkimaulana/secrets/firebase/versions/latest';
    const [version] = await client.accessSecretVersion({name: secretName});
    const firebaseJson = version.payload.data.toString('utf8');
    return JSON.parse(firebaseJson); // Parse the JSON string into an object
}

async function init() {
    const server = Hapi.server({
        port: 3000,
        host: '0.0.0.0',
        routes: {
            cors: {
                origin: ['*'], // Allow all origins
                additionalHeaders: ['cache-control', 'x-requested-with']
            }
        }
    });

    const serviceAccount = await getFirebaseCredentials();
    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
        databaseURL: 'https://submissionmlgc-rifkimaulana.firebaseio.com'
    });

    const db = admin.firestore();
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
                output: 'stream',
                parse: true,
                multipart: true,
                allow: 'multipart/form-data',
                maxBytes: Number.MAX_SAFE_INTEGER,
            },
        },
        handler: async (request, h) => {
            const { image } = request.payload;
            if (!image || !image._data) {
                return h.response({
                    status: 'fail',
                    message: 'File is required',
                }).code(400);
            }

            if (image._data.length > 1000000) {
                return h.response({
                    status: 'fail',
                    message: 'Payload content length greater than maximum allowed: 1000000',
                }).code(413);
            }

            const filename = image.hapi.filename.toLowerCase();
            if (!filename.endsWith('.png')) {
                return h.response({
                    status: 'fail',
                    message: 'Terjadi kesalahan dalam melakukan prediksi',
                }).code(400);
            }

            try {
                const tensor = tf.node.decodeImage(image._data, 3)
                    .resizeBilinear([224, 224])
                    .expandDims(0)
                    .div(255.0);

                const prediction = model.predict(tensor).arraySync();
                const classification = prediction[0] > 0.58 ? 'Cancer' : 'Non-cancer';

                const predictionId = uuidv4();
                const predictionDoc = {
                    id: predictionId,
                    result: classification,
                    suggestion: classification === 'Cancer'
                        ? 'Segera periksa ke dokter!'
                        : 'Penyakit kanker tidak terdeteksi.',
                    createdAt: new Date().toISOString(),
                };

                await db.collection('predictions').doc(predictionId).set(predictionDoc);

                return h.response({
                    status: 'success',
                    message: 'Model is predicted successfully',
                    data: predictionDoc,
                }).code(201);
            } catch (error) {
                console.error(error);
                return h.response({
                    status: 'fail',
                    message: 'Terjadi kesalahan dalam melakukan prediksi',
                }).code(400);
            }
        },
    });

    await server.start();
    console.log('Server running on %s', server.info.uri);
}

process.on('unhandledRejection', (err) => {
    console.error(err);
    process.exit(1);
});

init();
