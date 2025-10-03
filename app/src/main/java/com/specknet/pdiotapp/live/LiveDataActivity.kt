package com.specknet.pdiotapp.live

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet
import com.specknet.pdiotapp.R
import com.specknet.pdiotapp.database.ActivityHistoryManager
import com.specknet.pdiotapp.database.ActivityLog
import com.specknet.pdiotapp.database.AppDatabase
import com.specknet.pdiotapp.utils.Constants
import com.specknet.pdiotapp.utils.RESpeckLiveData
import com.specknet.pdiotapp.utils.ThingyLiveData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class LiveDataActivity : AppCompatActivity() {
    private val MAX_DATA_SET_SIZE = 200

    // **1. Define Normalization Constants**
    // Mean values from training
    private val MEAN_X = -0.03325532f
    private val MEAN_Y = -0.59998163f
    private val MEAN_Z = 0.03538302f

    // Standard deviation values from training
    private val STD_X = 0.45624453f
    private val STD_Y = 0.54131043f
    private val STD_Z = 0.51403646f

    // Global graph variables
    lateinit var dataSet_res_accel_x: LineDataSet
    lateinit var dataSet_res_accel_y: LineDataSet
    lateinit var dataSet_res_accel_z: LineDataSet

    lateinit var dataSet_thingy_accel_x: LineDataSet
    lateinit var dataSet_thingy_accel_y: LineDataSet
    lateinit var dataSet_thingy_accel_z: LineDataSet

    var time = 0f
    lateinit var allRespeckData: LineData
    lateinit var allThingyData: LineData

    lateinit var historyManager: ActivityHistoryManager

    lateinit var respeckChart: LineChart
    lateinit var thingyChart: LineChart

    lateinit var wakefulTextView: TextView
    lateinit var physicalTextView: TextView
    lateinit var socialTextView: TextView

    // Global broadcast receiver so we can unregister it
    lateinit var respeckLiveUpdateReceiver: BroadcastReceiver
    lateinit var thingyLiveUpdateReceiver: BroadcastReceiver
    lateinit var looperRespeck: Looper
    lateinit var looperThingy: Looper

    lateinit var user_email: String

    val filterTestRespeck = IntentFilter(Constants.ACTION_RESPECK_LIVE_BROADCAST)
    val filterTestThingy = IntentFilter(Constants.ACTION_THINGY_BROADCAST)

    // Model interpreters
    lateinit var wakefulModel: Interpreter
    lateinit var physicalModel: Interpreter
    lateinit var socialModel: Interpreter

    val noToString: Map<Int, String> = mapOf(
        0 to "ascending",
        1 to "descending",
        2 to "lying on back",
        3 to "lying on left",
        4 to "lying on right",
        5 to "lying on stomach",
        6 to "misc",
        7 to "normal walking",
        8 to "running",
        9 to "shuffle walking",
        10 to "sitting / standing"
    )

    val UPDATE_FREQUENCY = 2000 // in milliseconds

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_live_data)

        val db = AppDatabase.getDatabase(applicationContext)
        historyManager = db.activityHistoryManager()

        user_email = intent.getStringExtra("user_email") ?: "No User"

        Log.d("THINGY LIVE", Constants.ACTION_RESPECK_LIVE_BROADCAST)
        Log.d("RESPECK LIVE", Constants.ACTION_THINGY_BROADCAST)

        socialTextView = findViewById(R.id.socialTextView)
        physicalTextView = findViewById(R.id.physicalTextView)
        wakefulTextView = findViewById(R.id.wakefulTextView)
        setupCharts()

        Log.d("Preload tensor", "got here")
        try {
            wakefulModel = Interpreter(loadModelFile("TRY_2.tflite"))
            physicalModel = Interpreter(loadModelFile("TRY_2.tflite"))
            socialModel = Interpreter(loadModelFile("TRY_2.tflite"))
        } catch (e: IOException) {
            Log.e("ModelError", "Error loading models", e)
        }
        Log.d("Postload tensor", "got here")

        val modelToTextBox: Map<Interpreter?, TextView> = mapOf(
            wakefulModel to wakefulTextView,
            physicalModel to physicalTextView,
            socialModel to socialTextView
        )

        val windowSize = 50  // Adjust based on model requirements

        respeckListener(windowSize, modelToTextBox)
        thingyListener()
    }

    @Throws(IOException::class)
    private fun loadModelFile(tflite: String): MappedByteBuffer {
        val MODEL_ASSETS_PATH = tflite
        val assetFileDescriptor = this.assets.openFd(MODEL_ASSETS_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // **2. Implement a Normalization Function**
    private fun normalize(x: Float, y: Float, z: Float): FloatArray {
        val normalizedX = (x - MEAN_X) / STD_X
        val normalizedY = (y - MEAN_Y) / STD_Y
        val normalizedZ = (z - MEAN_Z) / STD_Z
        return floatArrayOf(normalizedX, normalizedY, normalizedZ)
    }

    fun runModel(interpreter: Interpreter?, sensorData: Array<FloatArray>): String {
        // **3. Ensure Correct Input Shape**
        // The model expects input shape: [batch_size, window_size, channels]
        // Here, batch_size = 1
        val sensorDataShaped = arrayOf(sensorData) // Shape: [1, window_size, 3]

        Log.d("tensor shape", "${sensorDataShaped.size}, ${sensorDataShaped[0].size}, ${sensorDataShaped[0][0].size}")
        Log.d("Running model", "Running model")

        val outputArray = Array(1) { FloatArray(11) } // Adjusted to match [1, 11]
        interpreter?.run(sensorDataShaped, outputArray)

        val outNo = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
        val classification = noToString[outNo] ?: "Invalid activity number"

        // **Note:** The original runModel function inserts an empty activity log.
        // This might be redundant as the caller also inserts a log with the actual classification.
        // You may want to remove or adjust this based on your requirements.

        // Remove or comment out the following block if redundant:
        /*
        GlobalScope.launch(Dispatchers.IO) {
            val dateFormat = SimpleDateFormat("dd-MM-yyyy HH:mm:ss", Locale.getDefault())
            val timestamp = dateFormat.format(Date())
            historyManager.insert(ActivityLog(userEmail = user_email, activity = "", timeStamp = timestamp))
        }
        */

        return classification
    }

    fun respeckListener(windowSize: Int, modelViewMap: Map<Interpreter?, TextView>) {
        // Set up the broadcast receiver
        var lastExecutionTime = 0L
        respeckLiveUpdateReceiver = object : BroadcastReceiver() {
            val sensorDataBuffer = ArrayList<FloatArray>() // To store normalized sensor readings

            override fun onReceive(context: Context, intent: Intent) {
                Log.d("receives data", "receives data")
                if (intent.action == Constants.ACTION_RESPECK_LIVE_BROADCAST) {

                    val liveData = intent.getSerializableExtra(Constants.RESPECK_LIVE_DATA) as RESpeckLiveData

                    // Get all relevant intent contents
                    val x = liveData.accelX
                    val y = liveData.accelY
                    val z = liveData.accelZ

                    // **4. Normalize the sensor data**
                    val normalizedReading = normalize(x, y, z)

                    // Add normalized data to the buffer
                    sensorDataBuffer.add(normalizedReading)
                    Log.d("sensorDataBuffer", sensorDataBuffer.toString())
                    if (sensorDataBuffer.size > windowSize) {
                        Log.d("removing data", "removing data")
                        // **Change from removeFirst() to removeAt(0) for compatibility**
                        sensorDataBuffer.removeAt(0)
                    }
                    Log.d("window size", sensorDataBuffer.size.toString())
                    if (sensorDataBuffer.size == windowSize) {
                        Log.d("correct window size", "correct window size")
                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastExecutionTime >= UPDATE_FREQUENCY) {
                            lastExecutionTime = currentTime
                            runOnUiThread {
                                Log.d("UI thread", "UI Thread")
                                // Prepare the input data with normalized values
                                val inputData = arrayOf(sensorDataBuffer.toTypedArray()) // Shape: [1, window_size, 3]
                                // Run the model for each interpreter and update the corresponding TextView
                                modelViewMap.forEach { (model, textbox) ->
                                    val modelOutput = runModel(model, inputData)
                                    textbox.text = modelOutput
                                    GlobalScope.launch(Dispatchers.IO) {
                                        val dateFormat = SimpleDateFormat("dd-MM-yyyy HH:mm:ss", Locale.getDefault())
                                        val timestamp = dateFormat.format(Date())
                                        Log.d(
                                            "input data",
                                            "Timestamp: $timestamp Email: $user_email Activity: $modelOutput"
                                        )

                                        try {
                                            Log.d("Database entry", "Inserting activity log")
                                            historyManager.insert(
                                                ActivityLog(
                                                    userEmail = user_email,
                                                    activity = modelOutput,
                                                    timeStamp = timestamp
                                                )
                                            )
                                            Log.d("Database entry", "Activity log inserted")
                                        } catch (e: Exception) {
                                            Log.e("DatabaseError", "Error inserting activity log", e)
                                        }
                                    }
                                }
                            }
                        }
                    }

                    time += 1
                    updateGraph("respeck", x, y, z) // Visualizing raw data
                }
            }
        }

        // Register receiver on another thread
        val handlerThreadRespeck = HandlerThread("bgThreadRespeckLive")
        handlerThreadRespeck.start()
        looperRespeck = handlerThreadRespeck.looper
        val handlerRespeck = Handler(looperRespeck)
        this.registerReceiver(respeckLiveUpdateReceiver, filterTestRespeck, null, handlerRespeck)
    }

    fun thingyListener() {
        thingyLiveUpdateReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {

                Log.i("thread", "I am running on thread = " + Thread.currentThread().name)

                val action = intent.action

                if (action == Constants.ACTION_THINGY_BROADCAST) {

                    val liveData =
                        intent.getSerializableExtra(Constants.THINGY_LIVE_DATA) as ThingyLiveData
                    Log.d("Live", "onReceive: liveData = $liveData")

                    // Get all relevant intent contents
                    val x = liveData.accelX
                    val y = liveData.accelY
                    val z = liveData.accelZ

                    // **Optional: Normalize the sensor data if needed**
                    // If Thingy data is also used with the model, normalize it similarly
                    val normalizedReading = normalize(x, y, z)

                    time += 1
                    updateGraph("thingy", x, y, z) // Visualizing raw data

                    // If you plan to use Thingy data with the model, implement buffering and model inference here
                    // Example:
                    /*
                    sensorDataBufferThingy.add(normalizedReading)
                    if (sensorDataBufferThingy.size > windowSize) {
                        sensorDataBufferThingy.removeAt(0)
                    }
                    if (sensorDataBufferThingy.size == windowSize) {
                        // Similar model inference as respeckListener
                    }
                    */
                }
            }
        }
        // Register receiver on another thread
        val handlerThreadThingy = HandlerThread("bgThreadThingyLive")
        handlerThreadThingy.start()
        looperThingy = handlerThreadThingy.looper
        val handlerThingy = Handler(looperThingy)
        this.registerReceiver(thingyLiveUpdateReceiver, filterTestThingy, null, handlerThingy)
    }

    fun setupCharts() {
        respeckChart = findViewById(R.id.respeck_chart)
        thingyChart = findViewById(R.id.thingy_chart)

        // Respeck

        time = 0f
        val entries_res_accel_x = ArrayList<Entry>()
        val entries_res_accel_y = ArrayList<Entry>()
        val entries_res_accel_z = ArrayList<Entry>()

        dataSet_res_accel_x = LineDataSet(entries_res_accel_x, "Accel X")
        dataSet_res_accel_y = LineDataSet(entries_res_accel_y, "Accel Y")
        dataSet_res_accel_z = LineDataSet(entries_res_accel_z, "Accel Z")

        dataSet_res_accel_x.setDrawCircles(false)
        dataSet_res_accel_y.setDrawCircles(false)
        dataSet_res_accel_z.setDrawCircles(false)

        dataSet_res_accel_x.setColor(
            ContextCompat.getColor(
                this,
                R.color.red
            )
        )
        dataSet_res_accel_y.setColor(
            ContextCompat.getColor(
                this,
                R.color.green
            )
        )
        dataSet_res_accel_z.setColor(
            ContextCompat.getColor(
                this,
                R.color.blue
            )
        )

        val dataSetsRes = ArrayList<ILineDataSet>()
        dataSetsRes.add(dataSet_res_accel_x)
        dataSetsRes.add(dataSet_res_accel_y)
        dataSetsRes.add(dataSet_res_accel_z)

        allRespeckData = LineData(dataSetsRes)
        respeckChart.data = allRespeckData
        respeckChart.invalidate()

        // Thingy

        time = 0f
        val entries_thingy_accel_x = ArrayList<Entry>()
        val entries_thingy_accel_y = ArrayList<Entry>()
        val entries_thingy_accel_z = ArrayList<Entry>()

        dataSet_thingy_accel_x = LineDataSet(entries_thingy_accel_x, "Accel X")
        dataSet_thingy_accel_y = LineDataSet(entries_thingy_accel_y, "Accel Y")
        dataSet_thingy_accel_z = LineDataSet(entries_thingy_accel_z, "Accel Z")

        dataSet_thingy_accel_x.setDrawCircles(false)
        dataSet_thingy_accel_y.setDrawCircles(false)
        dataSet_thingy_accel_z.setDrawCircles(false)

        dataSet_thingy_accel_x.setColor(
            ContextCompat.getColor(
                this,
                R.color.red
            )
        )
        dataSet_thingy_accel_y.setColor(
            ContextCompat.getColor(
                this,
                R.color.green
            )
        )
        dataSet_thingy_accel_z.setColor(
            ContextCompat.getColor(
                this,
                R.color.blue
            )
        )

        val dataSetsThingy = ArrayList<ILineDataSet>()
        dataSetsThingy.add(dataSet_thingy_accel_x)
        dataSetsThingy.add(dataSet_thingy_accel_y)
        dataSetsThingy.add(dataSet_thingy_accel_z)

        allThingyData = LineData(dataSetsThingy)
        thingyChart.data = allThingyData
        thingyChart.invalidate()
    }

    fun updateGraph(graph: String, x: Float, y: Float, z: Float) {
        if (graph == "respeck") {
            dataSet_res_accel_x.addEntry(Entry(time, x))
            dataSet_res_accel_y.addEntry(Entry(time, y))
            dataSet_res_accel_z.addEntry(Entry(time, z))

            limitDataSetSize(dataSet_res_accel_x, MAX_DATA_SET_SIZE)
            limitDataSetSize(dataSet_res_accel_y, MAX_DATA_SET_SIZE)
            limitDataSetSize(dataSet_res_accel_z, MAX_DATA_SET_SIZE)

            runOnUiThread {
                allRespeckData.notifyDataChanged()
                respeckChart.notifyDataSetChanged()
                respeckChart.invalidate()
                respeckChart.setVisibleXRangeMaximum(150f)
                respeckChart.moveViewToX(respeckChart.lowestVisibleX + 40)
            }
        } else if (graph == "thingy") {
            dataSet_thingy_accel_x.addEntry(Entry(time, x))
            dataSet_thingy_accel_y.addEntry(Entry(time, y))
            dataSet_thingy_accel_z.addEntry(Entry(time, z))

            limitDataSetSize(dataSet_thingy_accel_x, MAX_DATA_SET_SIZE)
            limitDataSetSize(dataSet_thingy_accel_y, MAX_DATA_SET_SIZE)
            limitDataSetSize(dataSet_thingy_accel_z, MAX_DATA_SET_SIZE)

            runOnUiThread {
                allThingyData.notifyDataChanged()
                thingyChart.notifyDataSetChanged()
                thingyChart.invalidate()
                thingyChart.setVisibleXRangeMaximum(150f)
                thingyChart.moveViewToX(thingyChart.lowestVisibleX + 40)
            }
        }
    }

    private fun limitDataSetSize(dataSet: LineDataSet, maxSize: Int) {
        while (dataSet.entryCount > maxSize) {
            dataSet.removeFirst()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        unregisterReceiver(respeckLiveUpdateReceiver)
        unregisterReceiver(thingyLiveUpdateReceiver)
        looperRespeck.quit()
        looperThingy.quit()

        // Close interpreters to free resources
        wakefulModel.close()
        physicalModel.close()
        socialModel.close()
    }
}
