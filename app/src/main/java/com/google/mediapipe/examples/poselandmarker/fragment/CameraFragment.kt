/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker.fragment

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.google.mediapipe.examples.poselandmarker.PoseLandmarkerHelper
import com.google.mediapipe.examples.poselandmarker.MainViewModel
import com.google.mediapipe.examples.poselandmarker.R
import com.google.mediapipe.examples.poselandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.json.JSONObject
import org.opencv.android.OpenCVLoader

class CameraFragment : Fragment(), PoseLandmarkerHelper.LandmarkerListener {
    companion object {
        private const val TAG = "Pose Landmarker"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var poseLandmarkerHelper: PoseLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_BACK

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the PoseLandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if(this::poseLandmarkerHelper.isInitialized) {
                if (poseLandmarkerHelper.isClose()) {
                    poseLandmarkerHelper.setupPoseLandmarker()
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if(this::poseLandmarkerHelper.isInitialized) {
            viewModel.setMinPoseDetectionConfidence(poseLandmarkerHelper.minPoseDetectionConfidence)
            viewModel.setMinPoseTrackingConfidence(poseLandmarkerHelper.minPoseTrackingConfidence)
            viewModel.setMinPosePresenceConfidence(poseLandmarkerHelper.minPosePresenceConfidence)
            viewModel.setDelegate(poseLandmarkerHelper.currentDelegate)

            // Close the PoseLandmarkerHelper and release resources
            backgroundExecutor.execute { poseLandmarkerHelper.clearPoseLandmarker() }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // START OPENCV ENGINE
     if (!OpenCVLoader.initDebug()) {
         Log.e(TAG, "OpenCV initialization failed.")
     } else {
         Log.d(TAG, "OpenCV initialized successfully.")
     }
        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        // Create the PoseLandmarkerHelper that will handle the inference
        backgroundExecutor.execute {
            poseLandmarkerHelper = PoseLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minPoseDetectionConfidence = viewModel.currentMinPoseDetectionConfidence,
                minPoseTrackingConfidence = viewModel.currentMinPoseTrackingConfidence,
                minPosePresenceConfidence = viewModel.currentMinPosePresenceConfidence,
                currentDelegate = viewModel.currentDelegate,
                poseLandmarkerHelperListener = this
            )
        }

        // Attach listeners to UI control widgets
        initBottomSheetControls()
    
    // Load the database when the camera screen opens!
        poseDatabase = loadPoseDatabase()
        if (poseDatabase.isNotEmpty()) {
            // Grab the very first pose in your database to start
            currentTargetPoseName = poseDatabase.keys.first()
            currentTargetPose = poseDatabase[currentTargetPoseName]

            Toast.makeText(requireContext(), "Loaded 44 Poses! Try: $currentTargetPoseName", Toast.LENGTH_LONG).show()
        }
    }

    private fun initBottomSheetControls() {
        // init bottom sheet settings

        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPoseDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPoseTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPosePresenceConfidence
            )

        // When clicked, lower pose detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseDetectionConfidence >= 0.2) {
                poseLandmarkerHelper.minPoseDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseDetectionConfidence <= 0.8) {
                poseLandmarkerHelper.minPoseDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower pose tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseTrackingConfidence >= 0.2) {
                poseLandmarkerHelper.minPoseTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseTrackingConfidence <= 0.8) {
                poseLandmarkerHelper.minPoseTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower pose presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPosePresenceConfidence >= 0.2) {
                poseLandmarkerHelper.minPosePresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPosePresenceConfidence <= 0.8) {
                poseLandmarkerHelper.minPosePresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        poseLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "PoseLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.setSelection(
            viewModel.currentModel,
            false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?,
                    p1: View?,
                    p2: Int,
                    p3: Long
                ) {
                    poseLandmarkerHelper.currentModel = p2
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset Poselandmarker
    // helper.
    private fun updateControlsUi() {
        if(this::poseLandmarkerHelper.isInitialized) {
            fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPoseDetectionConfidence
                )
            fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPoseTrackingConfidence
                )
            fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPosePresenceConfidence
                )

            // Needs to be cleared instead of reinitialized because the GPU
            // delegate needs to be initialized on the thread using it when applicable
            backgroundExecutor.execute {
                poseLandmarkerHelper.clearPoseLandmarker()
                poseLandmarkerHelper.setupPoseLandmarker()
            }
            fragmentCameraBinding.overlay.clear()
        }
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        detectPose(image)
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

@androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
 private fun detectPose(imageProxy: ImageProxy) {
     if(this::poseLandmarkerHelper.isInitialized) {
         try {
             val isFront = cameraFacing == CameraSelector.LENS_FACING_FRONT
             poseLandmarkerHelper.detectLiveStream(imageProxy, isFrontCamera = isFront)
         } catch (e: Exception) {
             imageProxy.close()
         }
     } else {
         imageProxy.close()
     }
 }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after pose have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
// ---------------------------------------------------------
    // AI MATH ENGINE (DATABASE VERSION)
    // ---------------------------------------------------------
    
    private var poseDatabase: Map<String, List<android.graphics.PointF>> = emptyMap()
    private var currentTargetPoseName: String = ""
    private var currentTargetPose: List<android.graphics.PointF>? = null

    // Reads your poses.json file from the assets folder
    private fun loadPoseDatabase(): Map<String, List<android.graphics.PointF>> {
        val poseMap = mutableMapOf<String, List<android.graphics.PointF>>()
        try {
            val jsonString = requireContext().assets.open("poses.json").bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)

            val keys = jsonObject.keys()
            while (keys.hasNext()) {
                val poseName = keys.next()
                val pointsArray = jsonObject.getJSONArray(poseName)
                val pointList = mutableListOf<android.graphics.PointF>()

                for (i in 0 until pointsArray.length()) {
                    val pointObj = pointsArray.getJSONObject(i)
                    val x = pointObj.getDouble("x").toFloat()
                    val y = pointObj.getDouble("y").toFloat()
                    pointList.add(android.graphics.PointF(x, y))
                }
                poseMap[poseName] = pointList
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return poseMap
    }

    override fun onResults(
        resultBundle: PoseLandmarkerHelper.ResultBundle
    ) {
        val results = resultBundle.results
        
        activity?.runOnUiThread {
            val overlay = _fragmentCameraBinding?.overlay
            if (overlay != null) {
                // Pass the CURRENT target pose to the screen so it draws it
                overlay.targetPose = currentTargetPose
                
                try {
                    if (results.isNotEmpty()) {
                        val firstResult = results.first()
                        
                        // Extract Native Segmentation Mask securely
                        firstResult.segmentationMasks()?.let { optionalMasks ->
                            if (optionalMasks.isPresent && optionalMasks.get().isNotEmpty()) {
                                val mask = optionalMasks.get().first()
                                val byteBuffer = com.google.mediapipe.framework.image.ByteBufferExtractor.extract(mask)
                                val maskFloatBuffer = byteBuffer.asFloatBuffer()
                                overlay.setSegmentationMask(maskFloatBuffer, mask.width, mask.height)
                            }
                        }

                        if (firstResult.landmarks().isNotEmpty() && currentTargetPose != null) {
                            val liveLandmarks = firstResult.landmarks()[0]
                            
                            // Normalize the live camera body!
                            val normalizedLive = normalizeLandmarks(liveLandmarks)
                            
                            // Compare Live Body vs Target Body
                            var totalError = 0.0
                            val jointsToCheck = listOf(11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
                            var validJointsCount = 0
                            
                            for (i in jointsToCheck) {
                                // CRASH FIX: Ensure both lists actually contain this joint index before doing math
                                if (i < currentTargetPose!!.size && i < normalizedLive.size) {
                                    val distance = Math.sqrt(
                                        Math.pow((currentTargetPose!![i].x - normalizedLive[i].x).toDouble(), 2.0) + 
                                        Math.pow((currentTargetPose!![i].y - normalizedLive[i].y).toDouble(), 2.0)
                                    )
                                    totalError += distance
                                    validJointsCount++
                                }
                            }
                            
                            // Prevent division by zero if no joints were valid
                            if (validJointsCount > 0) {
                                val averageError = totalError / validJointsCount
                                // If error is low, the pose matches!
                                overlay.isPoseMatched = (averageError < 0.6) // Adjust 0.6 to make it harder or easier
                            } else {
                                overlay.isPoseMatched = false
                            }
                        } else {
                            overlay.isPoseMatched = false
                            overlay.clearLivePose()
                        }
                        
                        // Send the live skeleton fallback data to the overlay
                        overlay.setResults(
                            firstResult,
                            resultBundle.inputImageHeight,
                            resultBundle.inputImageWidth,
                            com.google.mediapipe.tasks.vision.core.RunningMode.LIVE_STREAM
                        )
                        
                    } else {
                        overlay.isPoseMatched = false
                        overlay.clearLivePose()
                    }
                } catch (e: Exception) {
                    // CATCH-ALL CRASH FIX: If the math fails, just mark the pose as false instead of closing the app!
                    overlay.isPoseMatched = false
                }
                
                overlay.invalidate()
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }

    private fun normalizeLandmarks(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): List<android.graphics.PointF> {
        val hipCenterX = (landmarks[23].x() + landmarks[24].x()) / 2f
        val hipCenterY = (landmarks[23].y() + landmarks[24].y()) / 2f
        val shoulderCenterX = (landmarks[11].x() + landmarks[12].x()) / 2f
        val shoulderCenterY = (landmarks[11].y() + landmarks[12].y()) / 2f

        val torsoScale = Math.sqrt(
            Math.pow((shoulderCenterX - hipCenterX).toDouble(), 2.0) +
            Math.pow((shoulderCenterY - hipCenterY).toDouble(), 2.0)
        ).toFloat()

        val safeScale = if (torsoScale < 0.001f) 1f else torsoScale

        return landmarks.map { lm ->
            android.graphics.PointF(
                (lm.x() - hipCenterX) / safeScale,
                (lm.y() - hipCenterY) / safeScale
            )
        }
    }
} // <-- THIS BRACKET CLOSES THE ENTIRE CameraFragment CLASS
