#!/usr/bin/env python3
"""
Comprehensive Image Search Analysis

Tests 20+ different image types and analyzes picture vs thumbnail vector performance.
This will help us understand how well our comprehensive search is working.

Usage:
    python scripts/comprehensive_image_analysis.py
"""

import base64
import io
import json
import statistics
import time
from typing import Dict, List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont


class ImageAnalyzer:
    """Comprehensive image search analyzer."""
    
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.results = []
        self.test_images = []
        
    def create_test_image(self, name: str, size: Tuple[int, int], color: str, 
                         shapes: List[str] = None, text: str = None) -> str:
        """Create a test image with specific characteristics."""
        img = Image.new('RGB', size, color)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        # Add text
        if text and font:
            draw.text((10, 10), text, fill="white", font=font)
            
        # Add shapes to make images distinctive
        if shapes:
            for shape in shapes:
                if shape == "circle":
                    draw.ellipse([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], 
                               outline="white", width=3)
                elif shape == "rectangle":
                    draw.rectangle([size[0]//6, size[1]//6, 5*size[0]//6, 5*size[1]//6], 
                                 outline="white", width=3)
                elif shape == "lines":
                    for i in range(0, size[0], 20):
                        draw.line([(i, 0), (i, size[1])], fill="white", width=2)
                elif shape == "cross":
                    draw.line([(size[0]//2, 0), (size[0]//2, size[1])], fill="white", width=4)
                    draw.line([(0, size[1]//2), (size[0], size[1]//2)], fill="white", width=4)
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')
    
    def generate_test_cases(self) -> List[Dict]:
        """Generate 20+ diverse test images."""
        test_cases = [
            # Size variations (anime posters are typically portrait)
            {"name": "small_portrait", "size": (100, 150), "color": "red", "shapes": ["circle"], "text": "ANIME"},
            {"name": "medium_portrait", "size": (200, 300), "color": "blue", "shapes": ["rectangle"], "text": "ACTION"},
            {"name": "large_portrait", "size": (300, 450), "color": "green", "shapes": ["cross"], "text": "DRAMA"},
            {"name": "small_landscape", "size": (150, 100), "color": "purple", "shapes": ["lines"], "text": "COMEDY"},
            {"name": "square_small", "size": (100, 100), "color": "orange", "shapes": ["circle"], "text": "MECHA"},
            {"name": "square_large", "size": (200, 200), "color": "cyan", "shapes": ["rectangle"], "text": "ROMANCE"},
            
            # Color variations (anime poster colors)
            {"name": "dark_poster", "size": (120, 180), "color": "black", "shapes": ["circle"], "text": "DARK"},
            {"name": "light_poster", "size": (120, 180), "color": "white", "shapes": ["rectangle"], "text": "LIGHT"},
            {"name": "red_action", "size": (120, 180), "color": "darkred", "shapes": ["cross"], "text": "ACTION"},
            {"name": "blue_sci_fi", "size": (120, 180), "color": "darkblue", "shapes": ["lines"], "text": "SCI-FI"},
            {"name": "pink_romance", "size": (120, 180), "color": "hotpink", "shapes": ["circle"], "text": "ROMANCE"},
            {"name": "gray_mecha", "size": (120, 180), "color": "gray", "shapes": ["rectangle"], "text": "MECHA"},
            {"name": "yellow_comedy", "size": (120, 180), "color": "gold", "shapes": ["cross"], "text": "COMEDY"},
            
            # Pattern variations
            {"name": "simple_minimal", "size": (120, 180), "color": "navy", "shapes": [], "text": "MINIMAL"},
            {"name": "complex_shapes", "size": (120, 180), "color": "maroon", "shapes": ["circle", "rectangle"], "text": "COMPLEX"},
            {"name": "line_pattern", "size": (120, 180), "color": "darkgreen", "shapes": ["lines"], "text": "PATTERN"},
            {"name": "cross_design", "size": (120, 180), "color": "indigo", "shapes": ["cross"], "text": "DESIGN"},
            
            # Genre-inspired designs
            {"name": "horror_black", "size": (120, 180), "color": "black", "shapes": ["cross"], "text": "HORROR"},
            {"name": "fantasy_purple", "size": (120, 180), "color": "purple", "shapes": ["circle"], "text": "FANTASY"},
            {"name": "sports_green", "size": (120, 180), "color": "forestgreen", "shapes": ["lines"], "text": "SPORTS"},
            {"name": "school_blue", "size": (120, 180), "color": "royalblue", "shapes": ["rectangle"], "text": "SCHOOL"},
            
            # Edge cases
            {"name": "tiny_image", "size": (50, 75), "color": "violet", "shapes": ["circle"], "text": "TINY"},
            {"name": "very_wide", "size": (300, 100), "color": "teal", "shapes": ["lines"], "text": "WIDE"},
            {"name": "very_tall", "size": (100, 400), "color": "salmon", "shapes": ["rectangle"], "text": "TALL"},
            {"name": "monochrome", "size": (120, 180), "color": "gray", "shapes": ["circle"], "text": "MONO"},
        ]
        
        print(f"📋 Generating {len(test_cases)} test images...")
        for i, case in enumerate(test_cases):
            print(f"   {i+1:2d}. Creating {case['name']} ({case['size'][0]}x{case['size'][1]}, {case['color']})")
            case['image_data'] = self.create_test_image(
                case['name'], case['size'], case['color'], 
                case.get('shapes', []), case.get('text', '')
            )
        
        return test_cases
    
    def search_image(self, image_data: str, limit: int = 10) -> Dict:
        """Search for anime using image data."""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_base}/api/search/by-image-base64",
                data={"image_data": image_data, "limit": limit},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "data": data,
                    "response_time": end_time - start_time,
                    "status_code": response.status_code
                }
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "response_time": end_time - start_time,
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0,
                "status_code": 0
            }
    
    def analyze_results(self, test_case: Dict, search_result: Dict) -> Dict:
        """Analyze search results for picture vs thumbnail performance."""
        analysis = {
            "test_name": test_case["name"],
            "test_size": test_case["size"],
            "test_color": test_case["color"],
            "success": search_result["success"],
            "response_time": search_result["response_time"],
            "total_results": 0,
            "picture_scores": [],
            "thumbnail_scores": [],
            "visual_scores": [],
            "picture_vs_thumbnail_ratio": 0,
            "dominant_vector": "none",
            "avg_picture_score": 0,
            "avg_thumbnail_score": 0,
            "avg_visual_score": 0,
            "results_with_picture_data": 0,
            "results_with_thumbnail_data": 0,
            "top_match": None
        }
        
        if not search_result["success"]:
            analysis["error"] = search_result.get("error", "Unknown error")
            return analysis
        
        results = search_result["data"].get("results", [])
        analysis["total_results"] = len(results)
        
        if not results:
            return analysis
        
        # Extract scores from all results
        for result in results:
            picture_score = result.get("picture_score", 0)
            thumbnail_score = result.get("thumbnail_score", 0)
            visual_score = result.get("visual_similarity_score", 0)
            
            if picture_score is not None and picture_score > 0:
                analysis["picture_scores"].append(picture_score)
                analysis["results_with_picture_data"] += 1
                
            if thumbnail_score is not None and thumbnail_score > 0:
                analysis["thumbnail_scores"].append(thumbnail_score)
                analysis["results_with_thumbnail_data"] += 1
                
            if visual_score is not None and visual_score > 0:
                analysis["visual_scores"].append(visual_score)
        
        # Calculate averages
        if analysis["picture_scores"]:
            analysis["avg_picture_score"] = statistics.mean(analysis["picture_scores"])
        if analysis["thumbnail_scores"]:
            analysis["avg_thumbnail_score"] = statistics.mean(analysis["thumbnail_scores"])
        if analysis["visual_scores"]:
            analysis["avg_visual_score"] = statistics.mean(analysis["visual_scores"])
        
        # Determine dominant vector type
        if analysis["avg_picture_score"] > analysis["avg_thumbnail_score"] * 1.5:
            analysis["dominant_vector"] = "picture"
        elif analysis["avg_thumbnail_score"] > analysis["avg_picture_score"] * 1.5:
            analysis["dominant_vector"] = "thumbnail"
        else:
            analysis["dominant_vector"] = "balanced"
        
        # Calculate ratio
        if analysis["avg_thumbnail_score"] > 0:
            analysis["picture_vs_thumbnail_ratio"] = analysis["avg_picture_score"] / analysis["avg_thumbnail_score"]
        else:
            analysis["picture_vs_thumbnail_ratio"] = float('inf') if analysis["avg_picture_score"] > 0 else 0
        
        # Top match details
        if results:
            top = results[0]
            analysis["top_match"] = {
                "title": top.get("title", "Unknown"),
                "picture_score": top.get("picture_score", 0),
                "thumbnail_score": top.get("thumbnail_score", 0),
                "visual_score": top.get("visual_similarity_score", 0),
                "tags": top.get("tags", [])[:5]  # First 5 tags
            }
        
        return analysis
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive analysis on all test cases."""
        print("🔬 Starting Comprehensive Image Search Analysis")
        print("=" * 60)
        
        # Generate test cases
        test_cases = self.generate_test_cases()
        
        print(f"\n🧪 Testing {len(test_cases)} different image configurations...")
        print("=" * 60)
        
        all_analyses = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📸 Test {i:2d}/{len(test_cases)}: {test_case['name']}")
            print(f"   📐 Size: {test_case['size'][0]}x{test_case['size'][1]}")
            print(f"   🎨 Color: {test_case['color']}")
            
            # Search for similar anime
            search_result = self.search_image(test_case['image_data'], limit=10)
            
            # Analyze results
            analysis = self.analyze_results(test_case, search_result)
            all_analyses.append(analysis)
            
            # Print immediate results
            if analysis["success"]:
                print(f"   ✅ Found {analysis['total_results']} results in {analysis['response_time']:.3f}s")
                print(f"   📊 Picture: {analysis['avg_picture_score']:.3f}, Thumbnail: {analysis['avg_thumbnail_score']:.3f}")
                print(f"   🎯 Dominant: {analysis['dominant_vector']}")
                
                if analysis["top_match"]:
                    top = analysis["top_match"]
                    print(f"   🥇 Top: {top['title']} (Visual: {top['visual_score']:.3f})")
            else:
                print(f"   ❌ Failed: {analysis.get('error', 'Unknown error')}")
        
        return self.generate_summary_report(all_analyses)
    
    def generate_summary_report(self, analyses: List[Dict]) -> Dict:
        """Generate comprehensive summary report."""
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 60)
        
        # Overall statistics
        successful_tests = [a for a in analyses if a["success"]]
        failed_tests = [a for a in analyses if not a["success"]]
        
        print(f"\n📊 Overall Statistics:")
        print(f"   ✅ Successful tests: {len(successful_tests)}/{len(analyses)}")
        print(f"   ❌ Failed tests: {len(failed_tests)}")
        
        if not successful_tests:
            print("   ⚠️  No successful tests to analyze")
            return {"success": False}
        
        # Performance analysis
        response_times = [a["response_time"] for a in successful_tests]
        total_results = [a["total_results"] for a in successful_tests]
        
        print(f"\n⚡ Performance Analysis:")
        print(f"   🏃 Avg response time: {statistics.mean(response_times):.3f}s")
        print(f"   🏃 Min response time: {min(response_times):.3f}s")
        print(f"   🏃 Max response time: {max(response_times):.3f}s")
        print(f"   📊 Avg results per query: {statistics.mean(total_results):.1f}")
        
        # Vector performance analysis
        picture_heavy_tests = [a for a in successful_tests if a["dominant_vector"] == "picture"]
        thumbnail_heavy_tests = [a for a in successful_tests if a["dominant_vector"] == "thumbnail"]
        balanced_tests = [a for a in successful_tests if a["dominant_vector"] == "balanced"]
        
        print(f"\n🖼️  Vector Performance Analysis:")
        print(f"   📷 Picture-dominant results: {len(picture_heavy_tests)} ({len(picture_heavy_tests)/len(successful_tests)*100:.1f}%)")
        print(f"   🖼️  Thumbnail-dominant results: {len(thumbnail_heavy_tests)} ({len(thumbnail_heavy_tests)/len(successful_tests)*100:.1f}%)")
        print(f"   ⚖️  Balanced results: {len(balanced_tests)} ({len(balanced_tests)/len(successful_tests)*100:.1f}%)")
        
        # Score analysis
        all_picture_scores = []
        all_thumbnail_scores = []
        all_visual_scores = []
        
        for analysis in successful_tests:
            all_picture_scores.extend(analysis["picture_scores"])
            all_thumbnail_scores.extend(analysis["thumbnail_scores"])
            all_visual_scores.extend(analysis["visual_scores"])
        
        print(f"\n📈 Score Distribution Analysis:")
        if all_picture_scores:
            print(f"   📷 Picture scores - Avg: {statistics.mean(all_picture_scores):.3f}, "
                  f"Min: {min(all_picture_scores):.3f}, Max: {max(all_picture_scores):.3f}")
        if all_thumbnail_scores:
            print(f"   🖼️  Thumbnail scores - Avg: {statistics.mean(all_thumbnail_scores):.3f}, "
                  f"Min: {min(all_thumbnail_scores):.3f}, Max: {max(all_thumbnail_scores):.3f}")
        if all_visual_scores:
            print(f"   🎯 Visual scores - Avg: {statistics.mean(all_visual_scores):.3f}, "
                  f"Min: {min(all_visual_scores):.3f}, Max: {max(all_visual_scores):.3f}")
        
        # Image size impact analysis
        size_analysis = {}
        for analysis in successful_tests:
            size_key = f"{analysis['test_size'][0]}x{analysis['test_size'][1]}"
            if size_key not in size_analysis:
                size_analysis[size_key] = []
            size_analysis[size_key].append(analysis)
        
        print(f"\n📐 Image Size Impact Analysis:")
        for size, analyses_for_size in size_analysis.items():
            avg_visual = statistics.mean([a["avg_visual_score"] for a in analyses_for_size if a["avg_visual_score"] > 0])
            avg_response = statistics.mean([a["response_time"] for a in analyses_for_size])
            dominant_vectors = [a["dominant_vector"] for a in analyses_for_size]
            picture_dominant = dominant_vectors.count("picture")
            print(f"   📏 {size}: Avg Visual Score: {avg_visual:.3f}, "
                  f"Avg Response: {avg_response:.3f}s, Picture-dominant: {picture_dominant}/{len(analyses_for_size)}")
        
        # Best and worst performing tests
        print(f"\n🏆 Best Performing Tests:")
        best_tests = sorted(successful_tests, key=lambda x: x["avg_visual_score"], reverse=True)[:5]
        for i, test in enumerate(best_tests, 1):
            print(f"   {i}. {test['test_name']}: Visual Score {test['avg_visual_score']:.3f} "
                  f"({test['dominant_vector']})")
        
        print(f"\n⚠️  Lowest Performing Tests:")
        worst_tests = sorted(successful_tests, key=lambda x: x["avg_visual_score"])[:5]
        for i, test in enumerate(worst_tests, 1):
            print(f"   {i}. {test['test_name']}: Visual Score {test['avg_visual_score']:.3f} "
                  f"({test['dominant_vector']})")
        
        # Recommendations
        print(f"\n💡 Analysis Insights:")
        
        # Picture vs thumbnail effectiveness
        picture_effectiveness = len(picture_heavy_tests) / len(successful_tests) * 100
        thumbnail_effectiveness = len(thumbnail_heavy_tests) / len(successful_tests) * 100
        
        if picture_effectiveness > 70:
            print(f"   📷 Picture vectors are highly effective ({picture_effectiveness:.1f}% dominant)")
        elif thumbnail_effectiveness > 30:
            print(f"   🖼️  Thumbnail vectors show significant contribution ({thumbnail_effectiveness:.1f}% dominant)")
        else:
            print(f"   ⚖️  Balanced vector performance - comprehensive search working well")
        
        # Performance insights
        if statistics.mean(response_times) < 0.1:
            print(f"   ⚡ Excellent performance - sub-100ms average response time")
        elif statistics.mean(response_times) < 0.5:
            print(f"   🏃 Good performance - sub-500ms average response time")
        else:
            print(f"   ⏰ Consider optimization - response times above 500ms")
        
        # Data coverage insights
        results_with_picture = sum(a["results_with_picture_data"] for a in successful_tests)
        results_with_thumbnail = sum(a["results_with_thumbnail_data"] for a in successful_tests)
        total_results_count = sum(a["total_results"] for a in successful_tests)
        
        print(f"   📊 Data coverage - Picture: {results_with_picture}/{total_results_count} "
              f"({results_with_picture/total_results_count*100:.1f}%), "
              f"Thumbnail: {results_with_thumbnail}/{total_results_count} "
              f"({results_with_thumbnail/total_results_count*100:.1f}%)")
        
        return {
            "success": True,
            "total_tests": len(analyses),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "avg_response_time": statistics.mean(response_times),
            "picture_dominant_percentage": picture_effectiveness,
            "thumbnail_dominant_percentage": thumbnail_effectiveness,
            "avg_visual_score": statistics.mean(all_visual_scores) if all_visual_scores else 0,
            "detailed_analyses": analyses
        }


def main():
    """Run the comprehensive image analysis."""
    analyzer = ImageAnalyzer()
    
    # Test service health first
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ FastAPI server not healthy. Please start it first:")
            print("   python -m src.main")
            return False
    except Exception:
        print("❌ FastAPI server not reachable. Please start it first:")
        print("   python -m src.main")
        return False
    
    # Run comprehensive analysis
    try:
        results = analyzer.run_comprehensive_analysis()
        
        if results["success"]:
            print(f"\n🎉 ANALYSIS COMPLETE!")
            print(f"✅ Successfully analyzed {results['successful_tests']} image search scenarios")
            print(f"📊 Average visual similarity score: {results['avg_visual_score']:.3f}")
            print(f"⚡ Average response time: {results['avg_response_time']:.3f}s")
            print(f"📷 Picture vector dominance: {results['picture_dominant_percentage']:.1f}%")
            print(f"🖼️  Thumbnail vector contribution: {results['thumbnail_dominant_percentage']:.1f}%")
        else:
            print(f"\n❌ Analysis failed - no successful tests")
        
        return results["success"]
        
    except Exception as e:
        print(f"\n💥 Analysis failed with error: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        exit(1)