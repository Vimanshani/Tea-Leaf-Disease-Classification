import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';

class GeminiService {
  // Replace with your actual API key from aistudio.google.com/apikey
  static const String _apiKey = 'YOUR_API_KEY_HERE';

  static const String _baseUrl =
      'https://generativelanguage.googleapis.com/v1beta/models/'
      'gemini-2.5-flash:generateContent';

  /// Sends the detected disease name and confidence to Gemini
  /// and returns AI-generated treatment advice as plain text.
  static Future<String> getDiseaseAdvice({
  required String diseaseName,
  required double confidence,
}) async {
  final uri = Uri.parse('$_baseUrl?key=$_apiKey');

  final isHealthy = diseaseName.toLowerCase() == 'healthy';

  final prompt = isHealthy
      ? '''
A tea leaf has been scanned and classified as HEALTHY 
with ${confidence.toStringAsFixed(1)}% confidence by an AI model.

Provide a short, friendly response for a Sri Lankan tea farmer with:
1. A brief reassuring note (1 sentence)
2. Three practical tips to keep the plant healthy

Keep it concise, practical, and easy to understand.
Do not use markdown formatting, asterisks, or headers.
Use plain text with simple line breaks only.
'''
      : '''
A tea leaf disease detection AI model has identified the disease 
"$diseaseName" with ${confidence.toStringAsFixed(1)}% confidence.

Provide practical information for a Sri Lankan tea farmer including:
1. A brief description of this disease (1-2 sentences)
2. Three immediate treatment steps
3. Two prevention measures for the future

Keep the response concise and practical for field use.
Do not use markdown formatting, asterisks, or headers.
Use plain text with simple line breaks between sections.
''';

  try {
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'contents': [
          {
            'parts': [
              {'text': prompt}
            ]
          }
        ],
        'generationConfig': {
          'temperature': 0.4,
          'maxOutputTokens': 1024,
        }
      }),
    );

    debugPrint('Gemini status code: ${response.statusCode}');
    debugPrint('Gemini raw response: ${response.body}');

    if (response.statusCode == 200) {
      final decoded = jsonDecode(response.body);

      final candidates = decoded['candidates'];
      if (candidates == null || candidates.isEmpty) {
        return 'No advice available right now. Please try again.';
      }

      final finishReason = candidates[0]['finishReason'];
      debugPrint('Gemini finishReason: $finishReason');

      final content = candidates[0]['content'];
      if (content == null || content['parts'] == null) {
        return 'Received an incomplete response. Please try again.';
      }

      final parts = content['parts'] as List;
      final fullText = parts.map((p) => p['text'] ?? '').join('');

      if (fullText.trim().isEmpty) {
        return 'No advice text was returned. Please try again.';
      }

      return fullText;
    } else {
      debugPrint('Gemini error body: ${response.body}');
      return 'Could not fetch advice right now (status ${response.statusCode}). '
          'Please check your internet connection and try again.';
    }
  } catch (e) {
    debugPrint('Gemini exception: $e');
    return 'Unable to connect to AI advisor. '
        'Please check your internet connection.\n\nError: $e';
  }
}
}