#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Road Detection Pipeline
지도 내 핵심 도로 좌표 추정 시스템
"""

import streamlit as st
import os
import sys
from pathlib import Path
import json
import tempfile
import shutil
from PIL import Image
import pandas as pd
import base64

# Add the current directory to Python path to find road_detection_pipeline.py
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Fixed paths
HIGHLIGHT_MODEL_PATH = "/data/yuho/1.2025etri/2.e2e/3.highlight_model/road_unet.pth_1027_v3"
DATABASE_PATH = "/data/yuho/1.2025etri/2.e2e/2.korea_name_database/poi_all_filtered_300.parquet"

# Check if road_detection_pipeline.py exists in the same directory
pipeline_file = current_dir / "road_detection_pipeline.py"
if not pipeline_file.exists():
    st.error(f"""
    ❌ **파일을 찾을 수 없습니다: road_detection_pipeline.py**
    
    **필요한 파일 구조:**
    ```
    {current_dir}/
    ├── run.py (이 파일)
    ├── road_detection_pipeline.py (필요!)
    └── korean_place_names.parquet (데이터베이스)
    ```
    
    **현재 디렉토리:** `{current_dir}`
    
    road_detection_pipeline.py 파일을 이 디렉토리에 복사해주세요.
    """)
    st.stop()

# Try to import the pipeline module
try:
    from road_detection_pipeline import RoadDetectionPipeline
    PIPELINE_AVAILABLE = True
except Exception as e:
    st.error(f"""
    ❌ **road_detection_pipeline.py 모듈을 불러올 수 없습니다**
    
    **오류:** {str(e)}
    
    필요한 패키지가 모두 설치되어 있는지 확인하세요:
    - PaddleOCR
    - PyTorch
    - Transformers
    - OpenCV
    - pyproj
    - 등등
    """)
    PIPELINE_AVAILABLE = False
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="지도 도로 좌표 추정 시스템",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .status-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-error {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-info {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'temp_image_path' not in st.session_state:
    st.session_state.temp_image_path = None


def get_image_base64(image_path):
    """Convert image to base64 for display"""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None


def display_image_pair(img1_path, img2_path, title1="원본 지도", title2="결과 지도"):
    """Display two images side by side with reduced size"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {title1}")
        if os.path.exists(img1_path):
            st.image(img1_path, width=400)
        else:
            st.warning(f"이미지를 찾을 수 없습니다: {img1_path}")
    
    with col2:
        st.markdown(f"### {title2}")
        if os.path.exists(img2_path):
            st.image(img2_path, width=400)
        else:
            st.warning(f"이미지를 찾을 수 없습니다: {img2_path}")


def display_json_data(json_path, title="데이터"):
    """Display JSON data in expandable section"""
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with st.expander(f"📄 {title} (JSON)"):
            st.json(data)
    else:
        st.warning(f"파일을 찾을 수 없습니다: {json_path}")


def display_stage1_results(pipeline):
    """Display Stage 1 results"""
    st.markdown('<div class="section-header">📍 Stage 1: OCR → POI 추출 → 바운딩 박스</div>', unsafe_allow_html=True)
    
    result = st.session_state.result
    stage1 = result.get('stage1', {})
    
    # Check if we have stage1 data or just extraction_data
    if not stage1 and result.get('extraction_data'):
        st.warning("⚠️ Stage 1이 완료되지 않았지만 일부 중간 결과가 있습니다.")
        extraction_data = result.get('extraction_data', {})
        summary = extraction_data.get('summary', {})
        st.write(f"**총 추출:** {summary.get('total_extractions', 0)}개")
        st.write(f"**DB에서 발견:** {summary.get('found_in_db', 0)}개")
        st.write(f"**미발견:** {summary.get('not_found', 0)}개")
        st.info("💡 바운딩 박스 추출에 실패했습니다. 최소 3개 이상의 확인된 POI가 필요합니다.")
        return
    
    if not stage1:
        st.warning("Stage 1 결과가 없습니다.")
        return
    
    # Summary
    phase_num = stage1.get('phase', 'N/A')
    st.info(f"**완료 단계:** Phase {phase_num}")
    
    if stage1.get('status') == 'success':
        st.success("✅ Stage 1 완료: 바운딩 박스 추출 성공")
    else:
        st.warning(f"⚠️ Stage 1 상태: {stage1.get('status', 'unknown')}")
    
    # Phase tabs
    phase_tabs = st.tabs([
        "Phase 1: 초기 OCR",
        "Phase 2: 전체 LLM 수정",
        "Phase 3: 개별 LLM 수정",
        "Phase 4: 바운딩 박스"
    ])
    
    # Phase 1
    with phase_tabs[0]:
        st.markdown("#### Phase 1: 초기 OCR 추출")
        
        phase1_subtabs = st.tabs([
            "1a. 초기 OCR",
            "1b. Crop OCR",
            "1c. 규칙 필터링",
            "1d. LLM 필터링",
            "1e. LLM 랭킹",
            "1f. DB 검색"
        ])
        
        base_dir = pipeline.dirs['stage1_phase1']
        image_name = pipeline.image_name
        
        with phase1_subtabs[0]:
            st.markdown("##### 1a. 초기 OCR 추출")
            json_path = base_dir / f"{image_name}_1a_initial_ocr.json"
            viz_path = pipeline.dirs['visualizations'] / f"{image_name}_1a_initial_ocr.jpg"
            
            if os.path.exists(viz_path):
                st.image(viz_path, caption="OCR 추출 결과", width=600)
            display_json_data(json_path, "OCR 데이터")
        
        with phase1_subtabs[1]:
            st.markdown("##### 1b. Crop-level OCR")
            json_path = base_dir / f"{image_name}_1b_crop_ocr.json"
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                st.write(f"**처리된 텍스트 수:** {len(data.get('extracted_texts', []))}")
                
                # Show sample crops
                crops_dir = pipeline.dirs['crops']
                crop_files = sorted(crops_dir.glob(f"{image_name}_index*_crop.jpg"))
                
                if crop_files:
                    st.write("**샘플 Crop 이미지:**")
                    cols = st.columns(5)
                    for idx, crop_file in enumerate(crop_files[:10]):
                        with cols[idx % 5]:
                            st.image(crop_file, caption=f"Index {idx+1}", width=150)
                
                display_json_data(json_path, "Crop OCR 데이터")
        
        with phase1_subtabs[2]:
            st.markdown("##### 1c. 규칙 기반 필터링")
            json_path = base_dir / f"{image_name}_1c_rule_filtered.json"
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                filtered_count = len(data.get('rule_filtered_indices', []))
                remaining_count = len(data.get('extracted_texts', []))
                
                st.write(f"**필터링된 항목:** {filtered_count}개")
                st.write(f"**남은 항목:** {remaining_count}개")
                
                display_json_data(json_path, "규칙 필터링 데이터")
        
        with phase1_subtabs[3]:
            st.markdown("##### 1d. LLM 기반 필터링")
            json_path = base_dir / f"{image_name}_1d_llm_filtered.json"
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                filtered_count = len(data.get('llm_filtered_indices', []))
                remaining_count = len(data.get('extracted_texts', []))
                
                st.write(f"**LLM 필터링 항목:** {filtered_count}개")
                st.write(f"**남은 항목:** {remaining_count}개")
                
                display_json_data(json_path, "LLM 필터링 데이터")
        
        with phase1_subtabs[4]:
            st.markdown("##### 1e. LLM 랭킹")
            json_path = base_dir / f"{image_name}_1e_ranked.json"
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Show top ranked items
                texts = data.get('extracted_texts', [])
                sorted_texts = sorted(texts, key=lambda x: x.get('usefulness_rank', 999))
                
                st.write("**상위 10개 유용한 POI:**")
                for idx, item in enumerate(sorted_texts[:10], 1):
                    st.write(f"{idx}. {item.get('name', 'N/A')} (Rank: {item.get('usefulness_rank', 'N/A')})")
                
                display_json_data(json_path, "랭킹 데이터")
        
        with phase1_subtabs[5]:
            st.markdown("##### 1f. 데이터베이스 검색")
            json_path = base_dir / f"{image_name}_1f_db_search.json"
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                summary = data.get('summary', {})
                st.write(f"**총 추출:** {summary.get('total_extractions', 0)}개")
                st.write(f"**DB에서 발견:** {summary.get('found_in_db', 0)}개")
                st.write(f"**미발견:** {summary.get('not_found', 0)}개")
                
                display_json_data(json_path, "DB 검색 데이터")
    
    # Phase 2
    with phase_tabs[1]:
        st.markdown("#### Phase 2: 전체 LLM 수정")
        
        json_path = pipeline.dirs['stage1_phase2'] / f"{image_name}_2_all_in_one_revision.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            st.write(f"**총 추출:** {summary.get('total_extractions', 0)}개")
            st.write(f"**DB에서 발견:** {summary.get('found_in_db', 0)}개")
            st.write(f"**미발견:** {summary.get('not_found', 0)}개")
            
            display_json_data(json_path, "전체 LLM 수정 데이터")
        else:
            st.info("Phase 2가 실행되지 않았습니다.")
    
    # Phase 3
    with phase_tabs[2]:
        st.markdown("#### Phase 3: 개별 LLM 수정")
        
        json_path = pipeline.dirs['stage1_phase3'] / f"{image_name}_3_individual_revision.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            st.write(f"**총 추출:** {summary.get('total_extractions', 0)}개")
            st.write(f"**DB에서 발견:** {summary.get('found_in_db', 0)}개")
            st.write(f"**미발견:** {summary.get('not_found', 0)}개")
            
            display_json_data(json_path, "개별 LLM 수정 데이터")
        else:
            st.info("Phase 3가 실행되지 않았습니다.")
    
    # Phase 4
    with phase_tabs[3]:
        st.markdown("#### Phase 4: 바운딩 박스 추출")
        
        json_path = pipeline.dirs['stage1_phase4'] / f"{image_name}_4_bbox_result.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                bbox_data = json.load(f)
            
            st.write(f"**전략:** {bbox_data.get('strategy', 'N/A')}")
            st.write(f"**시도 횟수:** {bbox_data.get('attempt', 'N/A')}")
            st.write(f"**확인된 POI 수:** {len(bbox_data.get('confirmed_pois', []))}")
            
            # Display VWorld images
            vworld_img = bbox_data.get('vworld_image_path')
            marked_img = bbox_data.get('vworld_marked_image_path')
            
            if vworld_img and marked_img:
                display_image_pair(vworld_img, marked_img, "VWorld 지도", "POI 마킹된 VWorld 지도")
            
            # Show confirmed POIs
            st.write("**확인된 POI 목록:**")
            for poi in bbox_data.get('confirmed_pois', []):
                st.write(f"- {poi.get('name', 'N/A')} (Index: {poi.get('index', 'N/A')})")
            
            display_json_data(json_path, "바운딩 박스 데이터")
        else:
            st.warning("바운딩 박스가 추출되지 않았습니다.")


def display_stage2_results(pipeline):
    """Display Stage 2 results"""
    st.markdown('<div class="section-header">🗺️ Stage 2: 지도 정렬</div>', unsafe_allow_html=True)
    
    result = st.session_state.result
    stage2 = result.get('stage2', {})
    
    method = stage2.get('method', 'unknown')
    
    if method == 'affine_transformation':
        st.info("**방법:** Affine Transformation (POI 기반)")
        
        json_path = pipeline.dirs['stage2'] / f"{pipeline.image_name}_stage2a_affine_transformation.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Display correspondence points
            st.write("**POI 대응점:**")
            for pt in data.get('correspondence_points', []):
                st.write(f"- {pt.get('poi_name', 'N/A')}: ({pt.get('gps', {}).get('lon', 0):.6f}, {pt.get('gps', {}).get('lat', 0):.6f})")
            
            # Display images
            outputs = data.get('outputs', {})
            
            tabs = st.tabs(["중첩 이미지", "왜곡된 입력", "비교"])
            
            with tabs[0]:
                overlapped = outputs.get('overlapped_image')
                if overlapped and os.path.exists(overlapped):
                    st.image(overlapped, caption="중첩된 지도", width=600)
            
            with tabs[1]:
                warped = outputs.get('warped_input_image')
                if warped and os.path.exists(warped):
                    st.image(warped, caption="왜곡된 입력 지도", width=600)
            
            with tabs[2]:
                comparison = outputs.get('comparison_image')
                if comparison and os.path.exists(comparison):
                    st.image(comparison, caption="VWorld vs 중첩 비교", width=700)
            
            display_json_data(json_path, "Affine 변환 데이터")
    
    elif method == 'feature_matching':
        st.info("**방법:** Feature Matching (고급 특징 매칭)")
        
        json_path = pipeline.dirs['stage2'] / f"{pipeline.image_name}_stage2b_feature_matching.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            best_match = data.get('best_match', {})
            
            st.write("**최적 매칭:**")
            st.write(f"- 위치: ({best_match.get('position', {}).get('x', 0)}, {best_match.get('position', {}).get('y', 0)})")
            st.write(f"- 종합 점수: {best_match.get('composite_score', 0):.4f}")
            
            st.write("**개별 점수:**")
            for metric, score in best_match.get('individual_scores', {}).items():
                st.write(f"  - {metric}: {score:.4f}")
            
            # Display images
            outputs = data.get('outputs', {})
            
            tabs = st.tabs(["중첩 이미지", "혼합 영역", "비교", "히트맵"])
            
            with tabs[0]:
                overlapped = outputs.get('overlapped_image')
                if overlapped and os.path.exists(overlapped):
                    st.image(overlapped, caption="중첩된 지도", width=600)
            
            with tabs[1]:
                blended = outputs.get('blended_region')
                if blended and os.path.exists(blended):
                    st.image(blended, caption="혼합된 영역", width=600)
            
            with tabs[2]:
                comparison = outputs.get('comparison_image')
                if comparison and os.path.exists(comparison):
                    st.image(comparison, caption="VWorld vs 중첩 비교", width=700)
            
            with tabs[3]:
                heatmap = outputs.get('heatmap')
                if heatmap and os.path.exists(heatmap):
                    st.image(heatmap, caption="유사도 히트맵", width=600)
            
            display_json_data(json_path, "특징 매칭 데이터")


def display_stage3_results(pipeline):
    """Display Stage 3 results"""
    st.markdown('<div class="section-header">🛣️ Stage 3: 하이라이트 추출</div>', unsafe_allow_html=True)
    
    result = st.session_state.result
    stage3 = result.get('stage3')
    
    if not stage3:
        st.info("Stage 3가 실행되지 않았습니다.")
        return
    
    st.write(f"**스켈레톤 픽셀 수:** {stage3.get('num_skeleton_pixels', 0)}")
    
    outputs = stage3.get('outputs', {})
    
    tabs = st.tabs(["원본 마스크", "오버레이", "스켈레톤", "스켈레톤 오버레이"])
    
    with tabs[0]:
        mask_path = outputs.get('raw_mask')
        if mask_path and os.path.exists(mask_path):
            st.image(mask_path, caption="원본 하이라이트 마스크", width=600)
    
    with tabs[1]:
        overlay_path = outputs.get('overlay')
        if overlay_path and os.path.exists(overlay_path):
            st.image(overlay_path, caption="하이라이트 오버레이", width=600)
    
    with tabs[2]:
        skeleton_path = outputs.get('skeleton')
        if skeleton_path and os.path.exists(skeleton_path):
            st.image(skeleton_path, caption="스켈레톤", width=600)
    
    with tabs[3]:
        skeleton_overlay_path = outputs.get('skeleton_overlay')
        if skeleton_overlay_path and os.path.exists(skeleton_overlay_path):
            st.image(skeleton_overlay_path, caption="스켈레톤 오버레이", width=600)
    
    json_path = pipeline.dirs['stage3'] / f"{pipeline.image_name}_stage3_highlight_extraction.json"
    display_json_data(json_path, "하이라이트 추출 데이터")


def display_stage4_results(pipeline):
    """Display Stage 4 results"""
    st.markdown('<div class="section-header">🎯 Stage 4: VWorld 매핑</div>', unsafe_allow_html=True)
    
    result = st.session_state.result
    stage4 = result.get('stage4')
    
    if not stage4:
        st.info("Stage 4가 실행되지 않았습니다.")
        return
    
    st.write(f"**입력 스켈레톤 픽셀:** {stage4.get('input_skeleton_pixels', 0)}")
    st.write(f"**VWorld 매핑된 픽셀:** {stage4.get('vworld_mapped_pixels', 0)}")
    
    outputs = stage4.get('outputs', {})
    
    tabs = st.tabs(["초기 입력", "초기 VWorld", "병렬 비교", "최종 하이라이트"])
    
    with tabs[0]:
        initial_input = outputs.get('initial_highlight_on_input')
        if initial_input and os.path.exists(initial_input):
            st.image(initial_input, caption="초기 마스크 (입력 지도)", width=600)
    
    with tabs[1]:
        initial_vworld = outputs.get('initial_highlight_on_vworld')
        if initial_vworld and os.path.exists(initial_vworld):
            st.image(initial_vworld, caption="초기 마스크 (VWorld)", width=600)
    
    with tabs[2]:
        sidebyside = outputs.get('initial_sidebyside')
        if sidebyside and os.path.exists(sidebyside):
            st.image(sidebyside, caption="병렬 비교 (입력 vs VWorld)", width=700)
    
    with tabs[3]:
        final_highlight = outputs.get('highlight_on_vworld')
        if final_highlight and os.path.exists(final_highlight):
            st.image(final_highlight, caption="최종 하이라이트 (VWorld)", width=600)
    
    json_path = pipeline.dirs['stage4'] / f"{pipeline.image_name}_stage4_highlight_mapping.json"
    display_json_data(json_path, "하이라이트 매핑 데이터")


def display_stage5_results(pipeline):
    """Display Stage 5 results"""
    st.markdown('<div class="section-header">📍 Stage 5: GPS 좌표 계산</div>', unsafe_allow_html=True)
    
    result = st.session_state.result
    stage5 = result.get('stage5')
    
    if not stage5:
        st.info("Stage 5가 실행되지 않았습니다.")
        return
    
    polyline = stage5.get('polyline', {})
    stats = stage5.get('statistics', {})
    bbox_data = stage5.get('bounding_box', {})
    
    st.write(f"**폴리라인 포인트 수:** {polyline.get('num_points', 0)}")
    st.write(f"**총 거리:** {stats.get('total_distance_km', 0):.2f} km")
    st.write(f"**바운딩 박스:**")
    st.write(f"  - 경도: {bbox_data.get('min_lon', 0):.6f} ~ {bbox_data.get('max_lon', 0):.6f}")
    st.write(f"  - 위도: {bbox_data.get('min_lat', 0):.6f} ~ {bbox_data.get('max_lat', 0):.6f}")
    
    # Display polyline coordinates
    with st.expander("🗺️ GPS 좌표 보기 (처음 20개)"):
        coords = polyline.get('coordinates', [])
        if coords:
            coord_df = pd.DataFrame(coords[:20], columns=['경도 (Longitude)', '위도 (Latitude)'])
            st.dataframe(coord_df, width="stretch")
            
            if len(coords) > 20:
                st.info(f"총 {len(coords)}개 포인트 중 처음 20개만 표시됩니다.")
    
    # Display images
    outputs = stage5.get('outputs', {})
    
    tabs = st.tabs(["얇은 선", "두꺼운 선", "병렬 비교 (얇은)", "병렬 비교 (두꺼운)"])
    
    with tabs[0]:
        thin = outputs.get('visualization_thin')
        if thin and os.path.exists(thin):
            st.image(thin, caption="얇은 폴리라인", width=600)
    
    with tabs[1]:
        thick = outputs.get('visualization_thick_smooth')
        if thick and os.path.exists(thick):
            st.image(thick, caption="두껍고 부드러운 폴리라인", width=600)
    
    with tabs[2]:
        sidebyside_thin = outputs.get('visualization_thin_sidebyside')
        if sidebyside_thin and os.path.exists(sidebyside_thin):
            st.image(sidebyside_thin, caption="병렬 비교 - 얇은 선", width=700)
    
    with tabs[3]:
        sidebyside_thick = outputs.get('visualization_thick_sidebyside')
        if sidebyside_thick and os.path.exists(sidebyside_thick):
            st.image(sidebyside_thick, caption="병렬 비교 - 두꺼운 선", width=700)
    
    # Download buttons
    st.markdown("### 📥 다운로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GeoJSON download
        geojson_path = pipeline.dirs['stage5'] / f"{pipeline.image_name}_polyline.geojson"
        if os.path.exists(geojson_path):
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = f.read()
            st.download_button(
                label="📍 GeoJSON 다운로드",
                data=geojson_data,
                file_name=f"{pipeline.image_name}_polyline.geojson",
                mime="application/json"
            )
    
    with col2:
        # JSON download
        json_path = pipeline.dirs['stage5'] / f"{pipeline.image_name}_stage5_gps_polyline.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = f.read()
            st.download_button(
                label="📄 JSON 데이터 다운로드",
                data=json_data,
                file_name=f"{pipeline.image_name}_stage5_gps_polyline.json",
                mime="application/json"
            )
    
    display_json_data(json_path, "GPS 폴리라인 데이터")


def run_pipeline(image_file, llm_type, openai_key, vworld_key, database_path, highlight_model, matching_method):
    """Run the road detection pipeline"""
    # Import here to avoid circular dependency
    from road_detection_pipeline import RoadDetectionPipeline
    
    # Save uploaded file to temporary location
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, image_file.name)
    
    with open(temp_image_path, 'wb') as f:
        f.write(image_file.getbuffer())
    
    st.session_state.temp_image_path = temp_image_path
    
    # Initialize pipeline
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("파이프라인 초기화 중...")
        progress_bar.progress(10)
        
        pipeline_kwargs = {
            'output_dir': './road_detection_result',
            'database_path': database_path,
            'vworld_api_key': vworld_key if vworld_key else None,
            'llm_type': 'gpt' if llm_type == 'GPT-5' else 'qwen',
            'matching_method': matching_method
        }
        
        if llm_type == 'GPT-5':
            pipeline_kwargs['gpt_api_key'] = openai_key
            pipeline_kwargs['reasoning_effort'] = 'minimal'
        
        pipeline = RoadDetectionPipeline(**pipeline_kwargs)
        st.session_state.pipeline = pipeline
        
        progress_bar.progress(20)
        status_text.text("분석 실행 중... (수 분 소요될 수 있습니다)")
        
        # Run pipeline
        result = pipeline.run(temp_image_path, highlight_checkpoint=highlight_model)
        
        progress_bar.progress(100)
        
        # Always save result and pipeline for viewing intermediate results
        st.session_state.result = result
        st.session_state.pipeline = pipeline
        st.session_state.analysis_complete = True
        
        # Check actual status
        status = result.get('status', 'unknown')
        
        if status == 'success':
            status_text.text("분석 완료!")
            return True, "분석이 성공적으로 완료되었습니다!"
        elif status == 'insufficient_data':
            status_text.text("분석 불완전")
            message = result.get('message', '충분한 데이터를 찾지 못했습니다')
            return False, f"분석이 완전히 완료되지 않았습니다: {message}"
        else:
            status_text.text("분석 실패")
            message = result.get('message', '알 수 없는 오류')
            return False, f"분석에 실패했습니다: {message}"
        
    except Exception as e:
        # Even on exception, try to save whatever pipeline state exists
        if 'pipeline' in locals():
            st.session_state.pipeline = pipeline
            st.session_state.analysis_complete = True
        return False, f"오류 발생: {str(e)}"


def home_tab():
    """Home tab with input parameters"""
    st.markdown('<h1 class="main-title">🗺️ 지도 내 핵심 도로 좌표 추정 시스템</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="status-info">
    이 시스템은 지도 이미지에서 자동으로 도로 하이라이트를 추출하고 GPS 좌표로 변환합니다.
    </div>
    """, unsafe_allow_html=True)
    
    # Create 2-column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Section 1: LLM Selection
        st.markdown('<div class="section-header">1️⃣ LLM 선택</div>', unsafe_allow_html=True)
        
        llm_type = st.radio(
            "LLM 선택",
            options=['GPT-5', 'Qwen 3.5'],
            index=0,
            horizontal=True,
            help="GPT-5: OpenAI의 최신 모델 (API 키 필요)\nQwen 3.5: Alibaba의 오픈소스 VLM (로컬 실행)",
            label_visibility="collapsed"
        )
        
        # Section 2: API Keys
        st.markdown('<div class="section-header">2️⃣ API 키 입력</div>', unsafe_allow_html=True)
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="GPT-5 사용 시 필요합니다",
            disabled=(llm_type != 'GPT-5')
        )
        
        vworld_key = st.text_input(
            "VWorld API Key (선택사항)",
            type="password",
            help="VWorld 지도 API 키 (선택사항, 기본값 사용 가능)"
        )
        
        # Section 3: Additional Settings
        st.markdown('<div class="section-header">3️⃣ 추가 설정</div>', unsafe_allow_html=True)
        
        matching_method = st.selectbox(
            "지도 매칭 방법",
            options=['affine', 'feature'],
            index=0,
            help="affine: POI 기반 변환 (빠름)\nfeature: 고급 특징 매칭 (정확함)"
        )
        
        # Fixed paths info
        st.info(f"""
        **고정 경로:**
        - 데이터베이스: `{DATABASE_PATH}`
        - 하이라이트 모델: `{HIGHLIGHT_MODEL_PATH}`
        """)
        
        # Section 4: Image Upload
        st.markdown('<div class="section-header">4️⃣ 지도 업로드</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "지도 이미지 선택",
            type=['jpg', 'jpeg', 'png'],
            help="분석할 지도 이미지를 업로드하세요",
            label_visibility="collapsed"
        )
    
    with col_right:
        st.markdown('<div class="section-header">📷 업로드된 지도 미리보기</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="입력 지도", width=400)
        else:
            st.info("👈 왼쪽에서 지도 이미지를 업로드하세요")
    
    # Analysis Button (full width, below columns)
    st.markdown('<div class="section-header">5️⃣ 분석 시작</div>', unsafe_allow_html=True)
    
    # Validation
    can_analyze = True
    error_messages = []
    
    if uploaded_file is None:
        can_analyze = False
        error_messages.append("지도 이미지를 업로드해주세요.")
    
    if llm_type == 'GPT-5' and not openai_key:
        can_analyze = False
        error_messages.append("GPT-5 사용 시 OpenAI API Key가 필요합니다.")
    
    if not os.path.exists(DATABASE_PATH):
        can_analyze = False
        error_messages.append(f"데이터베이스 파일이 존재하지 않습니다: {DATABASE_PATH}")
    
    if not os.path.exists(HIGHLIGHT_MODEL_PATH):
        st.warning(f"⚠️ 하이라이트 모델을 찾을 수 없습니다: {HIGHLIGHT_MODEL_PATH}\nStage 1-2만 실행됩니다.")
        highlight_model = None
    else:
        highlight_model = HIGHLIGHT_MODEL_PATH
    
    if error_messages:
        for msg in error_messages:
            st.error(msg)
    
    analyze_button = st.button(
        "🚀 분석 시작",
        type="primary",
        disabled=not can_analyze,
        width="stretch"
    )
    
    if analyze_button and can_analyze:
        with st.spinner('분석 중... 잠시만 기다려주세요.'):
            success, message = run_pipeline(
                uploaded_file,
                llm_type,
                openai_key,
                vworld_key,
                DATABASE_PATH,
                highlight_model,
                matching_method
            )
            
            if success:
                st.success(message)
                st.balloons()
                st.info("📊 **결과 탭**으로 이동하여 분석 결과를 확인하세요!")
            else:
                st.error(message)
                st.warning("⚠️ 분석이 완전히 완료되지 않았습니다. **결과 탭**으로 이동하여 중간 결과 및 실패 이유를 확인하세요.")


def result_tab():
    """Result tab with analysis outputs"""
    if not st.session_state.analysis_complete:
        st.info("먼저 홈 탭에서 지도를 업로드하고 분석을 실행해주세요.")
        return
    
    result = st.session_state.result
    pipeline = st.session_state.pipeline
    
    # Show results even if not complete - as long as we have some data
    if not result and not pipeline:
        st.error("분석 결과를 불러올 수 없습니다.")
        return
    
    st.markdown('<h1 class="main-title">📊 분석 결과</h1>', unsafe_allow_html=True)
    
    # Status
    status = result.get('status', 'unknown') if result else 'error'
    
    if status == 'success':
        st.markdown('<div class="status-success">✅ 분석이 성공적으로 완료되었습니다!</div>', unsafe_allow_html=True)
    elif status == 'insufficient_data':
        st.markdown('<div class="status-error">⚠️ 분석이 불완전합니다 - 충분한 POI를 찾지 못했습니다</div>', unsafe_allow_html=True)
        if result:
            st.write(f"**상세:** {result.get('message', '')}")
        st.info("아래에서 중간 결과를 확인할 수 있습니다.")
    else:
        st.markdown(f'<div class="status-error">❌ 분석 실패: {status}</div>', unsafe_allow_html=True)
        if result:
            st.write(f"**상세:** {result.get('message', '')}")
        st.info("아래에서 사용 가능한 중간 결과를 확인할 수 있습니다.")
    
    # Main Results Section
    st.markdown('<div class="section-header">🎯 최종 결과</div>', unsafe_allow_html=True)
    
    # Show completion status for each stage
    if result:
        st.markdown("#### 📊 단계별 완료 상태")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if result.get('stage1'):
                st.success("✅ Stage 1")
            else:
                st.error("❌ Stage 1")
        
        with col2:
            if result.get('stage2'):
                st.success("✅ Stage 2")
            else:
                st.error("❌ Stage 2")
        
        with col3:
            if result.get('stage3'):
                st.success("✅ Stage 3")
            else:
                st.info("⊝ Stage 3")
        
        with col4:
            if result.get('stage4'):
                st.success("✅ Stage 4")
            else:
                st.info("⊝ Stage 4")
        
        with col5:
            if result.get('stage5'):
                st.success("✅ Stage 5")
            else:
                st.info("⊝ Stage 5")
        
        st.markdown("---")
    
    # Display input and final output side by side
    if status == 'success' and result:
        stage1 = result.get('stage1', {})
        bbox_result = stage1.get('bounding_box', {})
        
        original_img = bbox_result.get('original_image_path')
        
        # Find the best final output
        final_output = None
        
        # Try Stage 5 first (GPS with polyline)
        if result.get('stage5'):
            stage5_outputs = result['stage5'].get('outputs', {})
            final_output = stage5_outputs.get('visualization_thick_sidebyside') or \
                          stage5_outputs.get('visualization_thick_smooth')
        
        # Try Stage 4 (highlight mapping)
        if not final_output and result.get('stage4'):
            stage4_outputs = result['stage4'].get('outputs', {})
            final_output = stage4_outputs.get('initial_sidebyside') or \
                          stage4_outputs.get('highlight_on_vworld')
        
        # Try Stage 2 (map alignment)
        if not final_output:
            stage2_outputs = result['stage2'].get('outputs', {})
            final_output = stage2_outputs.get('comparison_image') or \
                          stage2_outputs.get('overlapped_image')
        
        if original_img and final_output:
            if os.path.exists(original_img) and os.path.exists(final_output):
                # If final output is side-by-side, show it full width
                if 'sidebyside' in str(final_output) or 'comparison' in str(final_output):
                    st.image(final_output, caption="최종 결과 (입력 vs 출력)", width=700)
                else:
                    display_image_pair(original_img, final_output, "원본 지도", "최종 결과")
    
    elif status == 'insufficient_data' and result:
        # Show what we extracted even if incomplete
        st.markdown("#### 추출된 POI 정보")
        
        extraction_data = result.get('extraction_data', {})
        if extraction_data:
            summary = extraction_data.get('summary', {})
            st.write(f"**총 추출:** {summary.get('total_extractions', 0)}개")
            st.write(f"**DB에서 발견:** {summary.get('found_in_db', 0)}개")
            st.write(f"**미발견:** {summary.get('not_found', 0)}개")
            
            if summary.get('found_in_db', 0) > 0:
                st.info("💡 아래 단계별 결과에서 추출된 POI 상세 정보를 확인할 수 있습니다.")
        else:
            st.warning("추출된 데이터가 없습니다.")
    
    else:
        # Unknown status or error - try to show whatever we have
        if pipeline:
            st.info("일부 중간 결과를 아래 단계별 결과에서 확인할 수 있습니다.")
    
    # GPS Polyline Display (if available)
    if status == 'success' and result.get('stage5'):
        st.markdown('<div class="section-header">📍 GPS 좌표 폴리라인</div>', unsafe_allow_html=True)
        
        stage5 = result['stage5']
        polyline = stage5.get('polyline', {})
        coords = polyline.get('coordinates', [])
        
        if coords:
            st.write(f"**총 포인트 수:** {len(coords)}")
            st.write(f"**총 거리:** {stage5['statistics']['total_distance_km']:.2f} km")
            
            # Display first and last few coordinates
            with st.expander("🗺️ GPS 좌표 시퀀스 (처음 10개 & 마지막 10개)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**시작 포인트**")
                    start_df = pd.DataFrame(coords[:10], columns=['경도', '위도'])
                    start_df.index = range(1, len(start_df) + 1)
                    st.dataframe(start_df, width="stretch")
                
                with col2:
                    st.markdown("**종료 포인트**")
                    end_df = pd.DataFrame(coords[-10:], columns=['경도', '위도'])
                    end_df.index = range(len(coords) - 9, len(coords) + 1)
                    st.dataframe(end_df, width="stretch")
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # GeoJSON
                geojson_path = pipeline.dirs['stage5'] / f"{pipeline.image_name}_polyline.geojson"
                if os.path.exists(geojson_path):
                    with open(geojson_path, 'r', encoding='utf-8') as f:
                        geojson_data = f.read()
                    st.download_button(
                        label="📍 GeoJSON 다운로드",
                        data=geojson_data,
                        file_name=f"{pipeline.image_name}_polyline.geojson",
                        mime="application/json",
                        width="stretch"
                    )
            
            with col2:
                # CSV
                coords_df = pd.DataFrame(coords, columns=['경도', '위도'])
                csv_data = coords_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📊 CSV 다운로드",
                    data=csv_data,
                    file_name=f"{pipeline.image_name}_coordinates.csv",
                    mime="text/csv",
                    width="stretch"
                )
            
            with col3:
                # Complete result JSON
                final_json_path = pipeline.dirs['root'] / f"{pipeline.image_name}_FINAL_RESULT.json"
                if os.path.exists(final_json_path):
                    with open(final_json_path, 'r', encoding='utf-8') as f:
                        final_json = f.read()
                    st.download_button(
                        label="📄 전체 결과 JSON",
                        data=final_json,
                        file_name=f"{pipeline.image_name}_FINAL_RESULT.json",
                        mime="application/json",
                        width="stretch"
                    )
    
    # Detailed Stage Results
    st.markdown('<div class="section-header">📂 단계별 상세 결과</div>', unsafe_allow_html=True)
    
    # Check what data we have available
    has_stage1 = result and result.get('stage1')
    has_extraction_data = result and result.get('extraction_data')
    has_stage2 = result and result.get('stage2')
    has_stage3 = result and result.get('stage3')
    has_stage4 = result and result.get('stage4')
    has_stage5 = result and result.get('stage5')
    
    # Build stage tabs based on available data
    stage_tabs = []
    
    if has_stage1 or has_extraction_data:
        stage_tabs.append("Stage 1: POI 추출")
    if has_stage2:
        stage_tabs.append("Stage 2: 지도 정렬")
    if has_stage3:
        stage_tabs.append("Stage 3: 하이라이트 추출")
    if has_stage4:
        stage_tabs.append("Stage 4: VWorld 매핑")
    if has_stage5:
        stage_tabs.append("Stage 5: GPS 계산")
    
    if not stage_tabs:
        st.warning("표시할 중간 결과가 없습니다.")
        return
    
    if stage_tabs:
        tabs = st.tabs(stage_tabs)
        
        tab_idx = 0
        
        if has_stage1 or has_extraction_data:
            with tabs[tab_idx]:
                if pipeline:
                    display_stage1_results(pipeline)
                else:
                    st.warning("파이프라인 데이터를 불러올 수 없습니다.")
            tab_idx += 1
        
        if has_stage2:
            with tabs[tab_idx]:
                if pipeline:
                    display_stage2_results(pipeline)
                else:
                    st.warning("파이프라인 데이터를 불러올 수 없습니다.")
            tab_idx += 1
        
        if has_stage3:
            with tabs[tab_idx]:
                if pipeline:
                    display_stage3_results(pipeline)
                else:
                    st.warning("파이프라인 데이터를 불러올 수 없습니다.")
            tab_idx += 1
        
        if has_stage4:
            with tabs[tab_idx]:
                if pipeline:
                    display_stage4_results(pipeline)
                else:
                    st.warning("파이프라인 데이터를 불러올 수 없습니다.")
            tab_idx += 1
        
        if has_stage5:
            with tabs[tab_idx]:
                if pipeline:
                    display_stage5_results(pipeline)
                else:
                    st.warning("파이프라인 데이터를 불러올 수 없습니다.")
    
    # Output directory info
    if pipeline:
        st.markdown('<div class="section-header">📁 출력 디렉토리</div>', unsafe_allow_html=True)
        st.info(f"모든 결과 파일은 다음 경로에 저장됩니다:\n`{pipeline.dirs['root']}`")


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🗺️ 지도 도로 좌표 추정")
        st.markdown("---")
        
        st.markdown("#### 📚 사용 가이드")
        st.markdown("""
        1. **홈 탭**에서 설정 및 분석 실행
        2. **결과 탭**에서 분석 결과 확인
        3. 각 단계별 상세 결과 탐색
        4. GPS 좌표 다운로드
        """)
        
        st.markdown("---")
        st.markdown("#### ℹ️ 시스템 정보")
        st.markdown("""
        - **버전:** 1.0.0
        - **개발:** Road Detection Pipeline
        - **파이프라인 단계:** 5단계
        """)
        
        if st.session_state.analysis_complete:
            st.markdown("---")
            st.success("✅ 분석 완료")
            
            if st.button("🔄 새 분석 시작", width="stretch"):
                # Reset session state
                st.session_state.analysis_complete = False
                st.session_state.result = None
                st.session_state.pipeline = None
                if st.session_state.temp_image_path:
                    try:
                        os.remove(st.session_state.temp_image_path)
                    except:
                        pass
                st.session_state.temp_image_path = None
                st.rerun()
    
    # Main tabs
    tab1, tab2 = st.tabs(["🏠 홈", "📊 결과"])
    
    with tab1:
        home_tab()
    
    with tab2:
        result_tab()


if __name__ == "__main__":
    main()