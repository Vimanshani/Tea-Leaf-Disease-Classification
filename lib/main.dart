import 'dart:io';
import 'dart:math';  
// import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'gemini_service.dart';

void main() {
  runApp(const TeaDiseaseApp());
}

// ─────────────────────────────────────────────
// APP ROOT
// ─────────────────────────────────────────────
class TeaDiseaseApp extends StatelessWidget {
  const TeaDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tea Leaf Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1B5E20),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        fontFamily: 'Roboto',
      ),
      home: const SplashScreen(),
    );
  }
}

// ─────────────────────────────────────────────
// SPLASH SCREEN
// ─────────────────────────────────────────────
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double>   _fadeAnim;
  late Animation<double>   _scaleAnim;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );
    _fadeAnim = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.0, 0.6, curve: Curves.easeIn),
      ),
    );
    _scaleAnim = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.elasticOut),
    );
    _controller.forward();

    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const HomePage()),
        );
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF1B5E20),
              Color(0xFF4CAF50),
              Color(0xFF81C784),
            ],
          ),
        ),
        child: Center(
          child: FadeTransition(
            opacity: _fadeAnim,
            child: ScaleTransition(
              scale: _scaleAnim,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.2),
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: Colors.white54,
                        width: 2,
                      ),
                    ),
                    child: const Icon(
                      Icons.eco,
                      size: 70,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 32),
                  const Text(
                    'Tea Leaf Detector',
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                      letterSpacing: 1.5,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'AI-powered tea disease detection',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.white.withValues(alpha: 0.85),
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 60),
                  SizedBox(
                    width: 40,
                    height: 40,
                    child: CircularProgressIndicator(
                      color: Colors.white.withValues(alpha: 0.8),
                      strokeWidth: 2,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
// HOME PAGE
// ─────────────────────────────────────────────
class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage>
    with SingleTickerProviderStateMixin {

  // ── Model — FIX: use ClassificationModel from pytorch_lite ──
  ClassificationModel? _model;
  List<String>         _labels        = [];
  
  String  _aiAdvice        = '';
  bool    _loadingAdvice   = false;

  // ── State ──
  File?   _image;
  String  _disease    = '';
  double  _confidence = 0.0;
  bool    _modelReady = false;
  bool    _analyzing  = false;

  // ── Animation ──
  late AnimationController _resultAnim;
  late Animation<Offset>   _slideAnim;
  late Animation<double>   _fadeResultAnim;

  @override
  void initState() {
    super.initState();
    _resultAnim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _slideAnim = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _resultAnim,
      curve: Curves.easeOutCubic,
    ));
    _fadeResultAnim =
        Tween<double>(begin: 0.0, end: 1.0).animate(_resultAnim);

    _loadModel();
  }

  @override
  void dispose() {
    _resultAnim.dispose();
    super.dispose();
  }

  // ─────────────────────────────────────────────
  // LOAD MODEL
  // FIX: uses pytorch_lite API — loadClassificationModel
  // instead of Interpreter.fromAsset (tflite_flutter)
  // ─────────────────────────────────────────────
  Future<void> _loadModel() async {
    debugPrint('=== STARTING MODEL LOAD ===');

    try {
      debugPrint('Step 1: Checking asset exists...');
      final byteData = await rootBundle.load('assets/mobilenetv2_tea_eca_noopt.ptl');
      debugPrint('Step 1 OK: .ptl file size = ${byteData.lengthInBytes} bytes');
    } catch (e) {
      debugPrint('Step 1 FAILED: .ptl file not found - $e');
      return;
    }

    try {
      debugPrint('Step 2: Checking labels.txt exists...');
      final labelStr = await rootBundle.loadString('assets/labels.txt');
      debugPrint('Step 2 OK: labels.txt content = $labelStr');
      _labels = labelStr.split('\n').where((l) => l.trim().isNotEmpty).toList();
      debugPrint('Step 2 OK: parsed ${_labels.length} labels');
    } catch (e) {
      debugPrint('Step 2 FAILED: labels.txt issue - $e');
      return;
    }

    try {
      debugPrint('Step 3: Loading PyTorch model...');
      _model = await PytorchLite.loadClassificationModel(
        'assets/mobilenetv2_tea_eca_noopt.ptl',
        224,
        224,
        8,
        labelPath: 'assets/labels.txt',
      );
      debugPrint('Step 3 OK: Model loaded successfully!');
    } catch (e, stack) {
      debugPrint('Step 3 FAILED: $e');
      debugPrint('Stack trace: $stack');
      return;
    }

    debugPrint('=== ALL STEPS COMPLETE — setting modelReady = true ===');
    setState(() => _modelReady = true);
  }

  // ── Pick from gallery ──
  Future<void> _fromGallery() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 100,
    );
    if (picked != null) {
      setState(() {
        _image      = File(picked.path);
        _disease    = '';
        _confidence = 0.0;
      });
      await _predict();
    }
  }

  // ── Capture from camera ──
  Future<void> _fromCamera() async {
    _showCameraHint();
  }

  void _showCameraHint() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        title: const Row(
          children: [
            Icon(Icons.tips_and_updates,
                 color: Color(0xFF2E7D32)),
            SizedBox(width: 8),
            Text('Photo Tips'),
          ],
        ),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _TipRow(
              icon: Icons.wb_sunny_outlined,
              text: 'Use good natural lighting',
            ),
            SizedBox(height: 8),
            _TipRow(
              icon: Icons.crop_free,
              text: 'Place leaf on plain light background',
            ),
            SizedBox(height: 8),
            _TipRow(
              icon: Icons.center_focus_strong,
              text: 'Fill frame with the leaf',
            ),
            SizedBox(height: 8),
            _TipRow(
              icon: Icons.do_not_touch,
              text: 'Keep camera steady',
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              final picker = ImagePicker();
              final picked = await picker.pickImage(
                source: ImageSource.camera,
                imageQuality: 100,
                preferredCameraDevice: CameraDevice.rear,
              );
              if (picked != null) {
                setState(() {
                  _image      = File(picked.path);
                  _disease    = '';
                  _confidence = 0.0;
                });
                await _predict();
              }
            },
            child: const Text(
              'Got it — open camera',
              style: TextStyle(
                color: Color(0xFF2E7D32),
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────
  // PREDICT
  // FIX: uses pytorch_lite getImagePrediction
  // instead of manual tensor building + Interpreter.run
  // pytorch_lite handles:
  //   - image resizing to 224x224
  //   - normalization with ImageNet mean/std
  //   - running inference
  //   - returning class probabilities
  // ─────────────────────────────────────────────
  Future<void> _predict() async {
    if (_model == null || _image == null) return;

    setState(() => _analyzing = true);
    _resultAnim.reset();

    try {
      final imageBytes = await _image!.readAsBytes();

      final List<double> rawOutput = await _model!.getImagePredictionList(
        imageBytes,
        mean: [0.485, 0.456, 0.406],
        std:  [0.229, 0.224, 0.225],
      );

      debugPrint('=== RAW LOGITS ===');
      for (int i = 0; i < rawOutput.length; i++) {
        debugPrint('  [$i] ${_labels.length > i ? _labels[i] : "?"}: ${rawOutput[i].toStringAsFixed(4)}');
      }

      final List<double> probs = _softmax(rawOutput);

      debugPrint('=== SOFTMAX PROBABILITIES ===');
      for (int i = 0; i < probs.length; i++) {
        debugPrint('  [$i] ${_labels.length > i ? _labels[i] : "?"}: ${probs[i].toStringAsFixed(4)}');
      }

      double maxProb = probs[0];
      int    maxIdx  = 0;
      for (int i = 1; i < probs.length; i++) {
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          maxIdx  = i;
        }
      }

      setState(() {
        _disease    = maxIdx < _labels.length ? _labels[maxIdx] : 'Unknown';
        _confidence = maxProb * 100;
        _analyzing  = false;
      });

      _resultAnim.forward();

      _fetchAdvice();

    } catch (e) {
      debugPrint('Inference error: $e');
      setState(() => _analyzing = false);
    }
  }
  
  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce((a, b) => a > b ? a : b);
    final exps     = logits.map((x) => exp(x - maxLogit)).toList();
    final sumExps  = exps.reduce((a, b) => a + b);
    return exps.map((x) => x / sumExps).toList();
  }

  Future<void> _fetchAdvice() async {
    setState(() {
     _loadingAdvice = true;
      _aiAdvice      = '';
    });

    final advice = await GeminiService.getDiseaseAdvice(
      diseaseName: _disease,
      confidence:  _confidence,
    );

    setState(() {
      _aiAdvice      = advice;
      _loadingAdvice = false;
    });
  }
  

  // ─────────────────────────────────────────────
  // BUILD UI
  // ─────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF9FBF9),
      body: CustomScrollView(
        slivers: [

          // ── App Bar ──
          SliverAppBar(
            expandedHeight: 120,
            pinned: true,
            flexibleSpace: FlexibleSpaceBar(
              title: const Text(
                'Tea Leaf Detector',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 20,
                ),
              ),
              background: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      Color(0xFF1B5E20),
                      Color(0xFF388E3C),
                    ],
                  ),
                ),
                child: Align(
                  alignment: Alignment.centerRight,
                  child: Padding(
                    padding: const EdgeInsets.only(
                        right: 16, top: 8),
                    child: Icon(
                      Icons.eco,
                      size: 60,
                      color: Colors.white.withValues(alpha: 0.15),
                    ),
                  ),
                ),
              ),
            ),
            actions: [
              Padding(
                padding: const EdgeInsets.all(16),
                child: _modelReady
                    ? const Icon(Icons.check_circle,
                                 color: Colors.white70,
                                 size: 20)
                    : const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2,
                        ),
                      ),
              ),
            ],
          ),

          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [

                  // ── Image preview ──
                  _buildImageCard(),
                  const SizedBox(height: 16),

                  // ── Buttons ──
                  _buildActionButtons(),
                  const SizedBox(height: 20),

                  // ── Analyzing ──
                  if (_analyzing) _buildAnalyzingCard(),

                  // ── Results ──
                  if (!_analyzing && _disease.isNotEmpty) ...[
                    FadeTransition(
                      opacity: _fadeResultAnim,
                      child: SlideTransition(
                        position: _slideAnim,
                        child: Column(
                          children: [
                            _buildResultCard(),
                            const SizedBox(height: 16),
                            _buildAdviceCard(),
                          ],
                        ),
                      ),
                    ),
                  ],

                  // ── Empty state ──
                  if (!_analyzing &&
                      _disease.isEmpty &&
                      _image == null)
                    _buildEmptyState(),

                  const SizedBox(height: 32),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────
  // WIDGETS
  // ─────────────────────────────────────────────

  Widget _buildImageCard() {
    return Card(
      elevation: 4,
      shadowColor: Colors.black26,
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20)),
      clipBehavior: Clip.antiAlias,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        height: _image != null ? 300 : 220,
        child: _image != null
            ? Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(_image!, fit: BoxFit.cover),
                  if (_analyzing)
                    Container(
                      color: Colors.black45,
                      child: const Center(
                        child: CircularProgressIndicator(
                            color: Colors.white),
                      ),
                    ),
                ],
              )
            : Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      const Color(0xFFE8F5E9),
                      const Color(0xFFC8E6C9).withValues(alpha: 0.5),
                    ],
                  ),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.add_photo_alternate_outlined,
                      size: 72,
                      color: const Color(0xFF4CAF50)
                          .withValues(alpha: 0.6),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Add a tea leaf photo',
                      style: TextStyle(
                        fontSize: 16,
                        color: const Color(0xFF2E7D32)
                            .withValues(alpha: 0.7),
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Use gallery or camera below',
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.grey[500],
                      ),
                    ),
                  ],
                ),
              ),
      ),
    );
  }

  Widget _buildActionButtons() {
    return Row(
      children: [
        Expanded(
          child: _ActionButton(
            icon:     Icons.photo_library_outlined,
            label:    'Gallery',
            subtitle: 'Choose photo',
            color:    const Color(0xFF2E7D32),
            onTap:    _modelReady ? _fromGallery : null,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _ActionButton(
            icon:     Icons.camera_alt_outlined,
            label:    'Camera',
            subtitle: 'Take photo',
            color:    const Color(0xFF1565C0),
            onTap:    _modelReady ? _fromCamera : null,
          ),
        ),
      ],
    );
  }

  Widget _buildAnalyzingCard() {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            const SizedBox(
              width: 28,
              height: 28,
              child: CircularProgressIndicator(
                strokeWidth: 3,
                color: Color(0xFF2E7D32),
              ),
            ),
            const SizedBox(width: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Analyzing leaf...',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                  ),
                ),
                Text(
                  'AI model is running',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[500],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
  final isHealthy = _disease.toLowerCase() == 'healthy';

  return Card(
    elevation: 4,
    shadowColor: Colors.black26,
    shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20)),
    child: Column(
      children: [
        // Colored header
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(
              horizontal: 20, vertical: 14),
          decoration: BoxDecoration(
            color: isHealthy
                ? const Color(0xFF2E7D32)
                : const Color(0xFFE65100).withValues(alpha: 0.12),
            borderRadius: const BorderRadius.only(
              topLeft:  Radius.circular(20),
              topRight: Radius.circular(20),
            ),
          ),
          child: Row(
            children: [
              Icon(
                isHealthy
                    ? Icons.check_circle
                    : Icons.warning_amber_rounded,
                color: isHealthy
                    ? Colors.white
                    : const Color(0xFFE65100),
                size: 24,
              ),
              const SizedBox(width: 10),
              Text(
                isHealthy
                    ? 'Healthy Leaf'
                    : 'Disease Detected',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 15,
                  color: isHealthy
                      ? Colors.white
                      : const Color(0xFFE65100),
                ),
              ),
            ],
          ),
        ),

        // Body
        Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                _disease,
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                  color: isHealthy
                      ? const Color(0xFF2E7D32)
                      : const Color(0xFF1A1A1A),
                ),
              ),
              const SizedBox(height: 16),

              // Confidence bar
              Row(
                children: [
                  Text(
                    'Confidence',
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.grey[600],
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    '${_confidence.toStringAsFixed(1)}%',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: _confidence > 80
                          ? const Color(0xFF2E7D32)
                          : const Color(0xFFE65100),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: LinearProgressIndicator(
                  value: _confidence / 100,
                  minHeight: 10,
                  backgroundColor: Colors.grey[200],
                  valueColor: AlwaysStoppedAnimation<Color>(
                    _confidence > 80
                        ? const Color(0xFF2E7D32)
                        : const Color(0xFFE65100),
                  ),
                ),
              ),

              if (_confidence < 70) ...[
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFF3E0),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                        color: const Color(0xFFFFB74D)),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.info_outline,
                           color: Color(0xFFE65100),
                           size: 18),
                      SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Low confidence. Try a clearer '
                          'photo with better lighting.',
                          style: TextStyle(
                            fontSize: 12,
                            color: Color(0xFFE65100),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ],
    ),
  );
 }

  

  Widget _buildEmptyState() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Column(
        children: [
          Icon(Icons.eco_outlined,
               size: 64, color: Colors.grey[300]),
          const SizedBox(height: 16),
          Text(
            'Ready to diagnose',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: Colors.grey[400],
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Take or choose a photo of a tea leaf\n'
            'to detect diseases instantly',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey[400],
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAdviceCard() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
             children: [
                Icon(Icons.auto_awesome,
                   color: Color(0xFF2E7D32), size: 20),
                SizedBox(width: 8),
                Text(
                  'AI Advisor',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1B5E20),
                  ),
                ),
              ],
            ),
            const Divider(height: 20),
            if (_loadingAdvice)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Row(
                  children: [
                    SizedBox(
                      width: 18, height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    SizedBox(width: 12),
                    Text('Getting advice...',
                      style: TextStyle(color: Colors.grey)),
                  ],
                ),
              )
            else
              Text(
                _aiAdvice,
                style: const TextStyle(
                  fontSize: 14,
                  height: 1.6,
                  color: Color(0xFF333333),
                ),
              ),
          ],
        ),
     ),
   );
  }
}

// ─────────────────────────────────────────────
// REUSABLE WIDGETS
// ─────────────────────────────────────────────

class _ActionButton extends StatelessWidget {
  final IconData      icon;
  final String        label;
  final String        subtitle;
  final Color         color;
  final VoidCallback? onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.color,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final disabled = onTap == null;
    return Material(
      color: disabled ? Colors.grey[200] : color,
      borderRadius: BorderRadius.circular(16),
      elevation: disabled ? 0 : 3,
      shadowColor: color.withValues(alpha: 0.4),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.symmetric(
              vertical: 16, horizontal: 12),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                icon,
                color: disabled ? Colors.grey : Colors.white,
                size: 26,
              ),
              const SizedBox(width: 10),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label,
                    style: TextStyle(
                      color: disabled
                          ? Colors.grey
                          : Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 15,
                    ),
                  ),
                  Text(
                    subtitle,
                    style: TextStyle(
                      color: disabled
                          ? Colors.grey[400]
                          : Colors.white70,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}



class _TipRow extends StatelessWidget {
  final IconData icon;
  final String   text;

  const _TipRow({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, color: const Color(0xFF2E7D32), size: 20),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            text,
            style: const TextStyle(fontSize: 13),
          ),
        ),
      ],
    );
  }
}
 