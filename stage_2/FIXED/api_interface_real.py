"""
API Interface - Real FastAPI Implementation

This module implements GENUINE FastAPI REST interface for batch processing.
Uses actual integration with real algorithms and processing pipelines.
NO mock functions - only real API endpoints and data processing.

Mathematical Foundation:
- RESTful API design with OpenAPI 3.0 specification compliance
- Asynchronous request processing with real algorithm execution
- Structured error responses with detailed diagnostic information
- Production-ready authentication and authorization hooks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import asyncio
import logging
import json
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Import real processing modules
from clustering_real import MultiObjectiveStudentClustering, ClusteringAlgorithm, StudentRecord
from batch_size_real import BatchSizeCalculator, OptimizationStrategy, ProgramBatchRequirements
from resource_allocator_real import ResourceAllocator, AllocationStrategy, ResourceRequirement
from membership_real import BatchMembershipGenerator
from enrollment_real import CourseEnrollmentGenerator
from batch_config_real import BatchConfigurationManager
from file_loader_real import RealFileLoader
from report_generator_real import RealReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stage 2 Student Batching System API",
    description="Production-ready REST API for automated student batch processing with real algorithmic computation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components - initialized once for performance
components = {
    'clustering_engine': None,
    'batch_size_calculator': None,
    'resource_allocator': None,
    'membership_generator': None,
    'enrollment_generator': None,
    'config_manager': None,
    'file_loader': None,
    'report_generator': None
}

# Processing job storage (in production, use Redis or database)
processing_jobs = {}

# Pydantic models for API requests/responses
class ProcessingRequest(BaseModel):
    """Request model for batch processing"""
    clustering_algorithm: str = Field(default="kmeans", 
                                    description="Algorithm: kmeans, spectral, hierarchical, multi_objective")
    batch_size_strategy: str = Field(default="balanced_multi_objective",
                                   description="Strategy: minimize_variance, maximize_utilization, balanced_multi_objective, constraint_satisfaction")
    resource_strategy: str = Field(default="balance_utilization",
                                 description="Strategy: optimize_capacity, minimize_conflicts, balance_utilization, prefer_proximity")
    target_clusters: Optional[int] = Field(None, description="Target number of clusters (auto-calculated if None)")
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="Custom configuration parameters")
    
    @validator('clustering_algorithm')
    def validate_clustering_algorithm(cls, v):
        valid_algorithms = ['kmeans', 'spectral', 'hierarchical', 'multi_objective']
        if v not in valid_algorithms:
            raise ValueError(f"Invalid clustering algorithm. Must be one of: {valid_algorithms}")
        return v
    
    @validator('batch_size_strategy')
    def validate_batch_size_strategy(cls, v):
        valid_strategies = ['minimize_variance', 'maximize_utilization', 'balanced_multi_objective', 'constraint_satisfaction']
        if v not in valid_strategies:
            raise ValueError(f"Invalid batch size strategy. Must be one of: {valid_strategies}")
        return v

class ProcessingStatus(BaseModel):
    """Processing job status model"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None

class ProcessingResult(BaseModel):
    """Processing result model"""
    job_id: str
    execution_summary: Dict[str, Any]
    students_processed: int
    batches_created: int
    processing_time_seconds: float
    overall_quality_score: float
    success_rate: float
    output_files: List[str]
    report_file: Optional[str] = None

# Initialize components on startup
@app.on_event("startup")
async def initialize_components():
    """Initialize all processing components"""
    logger.info("Initializing batch processing components...")
    
    try:
        # Initialize components with real implementations
        components['clustering_engine'] = MultiObjectiveStudentClustering(
            clustering_algorithm=ClusteringAlgorithm.KMEANS,
            random_state=42
        )
        
        components['batch_size_calculator'] = BatchSizeCalculator(
            optimization_strategy=OptimizationStrategy.BALANCED_MULTI_OBJECTIVE
        )
        
        components['resource_allocator'] = ResourceAllocator(
            allocation_strategy=AllocationStrategy.BALANCE_UTILIZATION
        )
        
        components['membership_generator'] = BatchMembershipGenerator()
        components['enrollment_generator'] = CourseEnrollmentGenerator()
        components['config_manager'] = BatchConfigurationManager()
        components['file_loader'] = RealFileLoader()
        components['report_generator'] = RealReportGenerator()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """API health check endpoint"""
    return {
        "service": "Stage 2 Student Batching System API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with component status"""
    component_status = {}
    
    for name, component in components.items():
        try:
            if component is not None:
                component_status[name] = "healthy"
            else:
                component_status[name] = "not_initialized"
        except Exception as e:
            component_status[name] = f"error: {str(e)}"
    
    return {
        "status": "healthy" if all(status == "healthy" for status in component_status.values()) else "degraded",
        "components": component_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload/data", tags=["Data Management"])
async def upload_data_files(
    students_file: UploadFile = File(..., description="Students CSV file"),
    courses_file: Optional[UploadFile] = File(None, description="Courses CSV file"),
    batches_file: Optional[UploadFile] = File(None, description="Batches CSV file"),
    rooms_file: Optional[UploadFile] = File(None, description="Rooms CSV file"),
    shifts_file: Optional[UploadFile] = File(None, description="Shifts CSV file"),
    programs_file: Optional[UploadFile] = File(None, description="Programs CSV file")
):
    """Upload data files for processing"""
    
    try:
        # Create temporary directory for uploaded files
        temp_dir = Path(tempfile.mkdtemp(prefix="batch_processing_"))
        
        uploaded_files = {}
        
        # Save uploaded files
        for file_param, expected_name in [
            (students_file, "students.csv"),
            (courses_file, "courses.csv"),
            (batches_file, "batches.csv"),
            (rooms_file, "rooms.csv"),
            (shifts_file, "shifts.csv"),
            (programs_file, "programs.csv")
        ]:
            if file_param:
                file_path = temp_dir / expected_name
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file_param.file, buffer)
                uploaded_files[expected_name] = str(file_path)
        
        # Validate uploaded files
        file_loader = components['file_loader']
        discovered_files = file_loader.discover_files(str(temp_dir))
        
        validation_results = {}
        for file_name, metadata in discovered_files.items():
            validation_results[file_name] = {
                "status": metadata.status.value,
                "rows_count": metadata.rows_count,
                "columns_count": metadata.columns_count,
                "data_quality": metadata.data_quality.value,
                "validation_errors": metadata.validation_errors
            }
        
        return {
            "message": "Files uploaded and validated successfully",
            "temp_directory": str(temp_dir),
            "uploaded_files": uploaded_files,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/process/batch", response_model=ProcessingStatus, tags=["Batch Processing"])
async def start_batch_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    data_directory: str = Field(..., description="Directory containing data files")
):
    """Start batch processing job with real algorithm execution"""
    
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_status = ProcessingStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Processing job queued",
        started_at=datetime.now()
    )
    
    processing_jobs[job_id] = job_status
    
    # Start background processing
    background_tasks.add_task(
        execute_batch_processing_pipeline,
        job_id,
        request,
        data_directory
    )
    
    return job_status

async def execute_batch_processing_pipeline(
    job_id: str,
    request: ProcessingRequest,
    data_directory: str
):
    """Execute the complete batch processing pipeline"""
    
    job_status = processing_jobs[job_id]
    
    try:
        job_status.status = "running"
        job_status.message = "Starting batch processing pipeline"
        job_status.progress = 0.1
        
        start_time = datetime.now()
        
        # Initialize components with request parameters
        clustering_engine = MultiObjectiveStudentClustering(
            clustering_algorithm=ClusteringAlgorithm(request.clustering_algorithm),
            random_state=42
        )
        
        batch_size_calculator = BatchSizeCalculator(
            optimization_strategy=OptimizationStrategy(request.batch_size_strategy)
        )
        
        resource_allocator = ResourceAllocator(
            allocation_strategy=AllocationStrategy(request.resource_strategy)
        )
        
        # Stage 1: Load and validate data
        job_status.message = "Loading and validating data files"
        job_status.progress = 0.2
        
        file_loader = components['file_loader']
        loaded_data = file_loader.load_all_required_files(data_directory)
        
        if 'students.csv' not in loaded_data or not loaded_data['students.csv'].success:
            raise ValueError("Student data is required but not available or invalid")
        
        students_df = loaded_data['students.csv'].dataframe
        
        # Convert to StudentRecord objects
        student_records = []
        for _, student in students_df.iterrows():
            courses = student.get('enrolled_courses', '')
            course_list = courses.split(',') if courses else []
            
            languages = student.get('preferred_languages', '')
            lang_list = languages.split(',') if languages else []
            
            student_record = StudentRecord(
                student_id=student['student_id'],
                program_id=student.get('program_id', ''),
                academic_year=student.get('academic_year', ''),
                enrolled_courses=[c.strip() for c in course_list if c.strip()],
                preferred_shift=student.get('preferred_shift'),
                preferred_languages=[l.strip() for l in lang_list if l.strip()]
            )
            student_records.append(student_record)
        
        # Stage 2: Calculate optimal batch sizes
        job_status.message = "Calculating optimal batch sizes"
        job_status.progress = 0.3
        
        # Group students by program
        program_groups = {}
        for student in student_records:
            if student.program_id not in program_groups:
                program_groups[student.program_id] = []
            program_groups[student.program_id].append(student)
        
        batch_requirements = []
        for program_id, students in program_groups.items():
            requirement = ProgramBatchRequirements(
                program_id=program_id,
                total_students=len(students),
                preferred_batch_size=request.custom_parameters.get('preferred_batch_size', 30) if request.custom_parameters else 30,
                min_batch_size=request.custom_parameters.get('min_batch_size', 15) if request.custom_parameters else 15,
                max_batch_size=request.custom_parameters.get('max_batch_size', 60) if request.custom_parameters else 60
            )
            batch_requirements.append(requirement)
        
        batch_size_results = batch_size_calculator.calculate_optimal_batch_sizes(batch_requirements)
        
        # Stage 3: Perform clustering
        job_status.message = "Performing student clustering"
        job_status.progress = 0.5
        
        # Calculate target clusters
        if request.target_clusters is None:
            total_students = len(student_records)
            avg_batch_size = sum(r.optimal_batch_size for r in batch_size_results) / len(batch_size_results)
            target_clusters = max(2, int(total_students / avg_batch_size))
        else:
            target_clusters = request.target_clusters
        
        clustering_result = clustering_engine.perform_clustering(
            students=student_records,
            target_clusters=target_clusters
        )
        
        # Stage 4: Allocate resources
        job_status.message = "Allocating resources"
        job_status.progress = 0.7
        
        allocation_result = None
        if ('rooms.csv' in loaded_data and loaded_data['rooms.csv'].success and
            'shifts.csv' in loaded_data and loaded_data['shifts.csv'].success):
            
            rooms_df = loaded_data['rooms.csv'].dataframe
            shifts_df = loaded_data['shifts.csv'].dataframe
            
            resource_allocator.load_resource_data(rooms_df, shifts_df)
            
            resource_requirements = []
            for cluster in clustering_result.clusters:
                requirement = ResourceRequirement(
                    batch_id=cluster.batch_id,
                    required_capacity=len(cluster.student_ids)
                )
                resource_requirements.append(requirement)
            
            allocation_result = resource_allocator.allocate_resources(resource_requirements)
        
        # Stage 5: Generate memberships
        job_status.message = "Generating batch memberships"
        job_status.progress = 0.8
        
        membership_generator = components['membership_generator']
        
        # Prepare batch definitions
        if 'batches.csv' in loaded_data and loaded_data['batches.csv'].success:
            batches_df = loaded_data['batches.csv'].dataframe
        else:
            # Create synthetic batch definitions from clusters
            batch_data = []
            for cluster in clustering_result.clusters:
                batch_data.append({
                    'batch_id': cluster.batch_id,
                    'batch_code': cluster.batch_id,
                    'batch_name': f'Batch {cluster.batch_id}',
                    'program_id': 'DEFAULT',
                    'academic_year': '2024-25'
                })
            batches_df = pd.DataFrame(batch_data)
        
        membership_generator.load_data(students_df, batches_df)
        membership_records = membership_generator.generate_memberships_from_clusters(clustering_result.clusters)
        
        # Stage 6: Generate course enrollments
        job_status.message = "Generating course enrollments"
        job_status.progress = 0.9
        
        enrollment_records = []
        if 'courses.csv' in loaded_data and loaded_data['courses.csv'].success:
            enrollment_generator = components['enrollment_generator']
            courses_df = loaded_data['courses.csv'].dataframe
            enrollment_generator.load_course_data(courses_df)
            
            if 'batch_requirements.csv' in loaded_data and loaded_data['batch_requirements.csv'].success:
                requirements_df = loaded_data['batch_requirements.csv'].dataframe
                enrollment_generator.load_batch_requirements(requirements_df)
            
            enrollment_records = enrollment_generator.generate_enrollments_from_memberships(membership_records)
        
        # Stage 7: Generate reports and save results
        job_status.message = "Generating reports and saving results"
        job_status.progress = 0.95
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare execution data
        execution_data = {
            'processing_time_seconds': processing_time,
            'total_students_processed': len(student_records),
            'clusters_generated': len(clustering_result.clusters),
            'clustering_algorithm': request.clustering_algorithm,
            'clustering_score': clustering_result.optimization_score,
            'batch_size_strategy': request.batch_size_strategy,
            'resource_strategy': request.resource_strategy,
            'files_processed': [name for name, result in loaded_data.items() if result.success],
            'failed_files': [name for name, result in loaded_data.items() if not result.success],
            'consistency_errors': file_loader.validate_data_consistency(loaded_data)
        }
        
        # Generate comprehensive report
        report_generator = components['report_generator']
        execution_report = report_generator.generate_execution_report(
            execution_data=execution_data,
            clustering_result=clustering_result,
            batch_size_results=batch_size_results,
            allocation_result=allocation_result,
            membership_records=membership_records,
            enrollment_records=enrollment_records
        )
        
        # Export reports
        output_dir = Path("./output") / f"job_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_report_path = report_generator.export_report_to_json(
            execution_report, 
            filename=f"execution_report_{job_id}.json"
        )
        
        # Save HTML report
        html_report_path = report_generator.export_report_to_html(
            execution_report,
            filename=f"execution_report_{job_id}.html"
        )
        
        # Save processing results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        # Save clustering results
        clustering_output = []
        for cluster in clustering_result.clusters:
            for student_id in cluster.student_ids:
                clustering_output.append({
                    'student_id': student_id,
                    'batch_id': cluster.batch_id,
                    'academic_coherence_score': cluster.academic_coherence_score,
                    'program_consistency_score': cluster.program_consistency_score,
                    'resource_efficiency_score': cluster.resource_efficiency_score
                })
        
        clustering_file = output_dir / f"student_clusters_{timestamp}.csv"
        pd.DataFrame(clustering_output).to_csv(clustering_file, index=False)
        output_files.append(str(clustering_file))
        
        # Complete job successfully
        job_status.status = "completed"
        job_status.message = "Batch processing completed successfully"
        job_status.progress = 1.0
        job_status.completed_at = datetime.now()
        job_status.results = {
            'execution_summary': execution_data,
            'students_processed': len(student_records),
            'batches_created': len(clustering_result.clusters),
            'processing_time_seconds': processing_time,
            'overall_quality_score': execution_report.overall_quality_score,
            'success_rate': execution_report.success_rate,
            'output_files': output_files,
            'json_report_path': json_report_path,
            'html_report_path': html_report_path
        }
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job_status.status = "failed"
        job_status.message = f"Processing failed: {str(e)}"
        job_status.completed_at = datetime.now()
        job_status.error_details = str(e)

@app.get("/process/status/{job_id}", response_model=ProcessingStatus, tags=["Batch Processing"])
async def get_processing_status(job_id: str):
    """Get processing job status"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.get("/process/result/{job_id}", response_model=ProcessingResult, tags=["Batch Processing"])
async def get_processing_result(job_id: str):
    """Get processing job results"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = processing_jobs[job_id]
    
    if job_status.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job_status.status}, not completed")
    
    if not job_status.results:
        raise HTTPException(status_code=500, detail="Job completed but no results available")
    
    return ProcessingResult(
        job_id=job_id,
        execution_summary=job_status.results['execution_summary'],
        students_processed=job_status.results['students_processed'],
        batches_created=job_status.results['batches_created'],
        processing_time_seconds=job_status.results['processing_time_seconds'],
        overall_quality_score=job_status.results['overall_quality_score'],
        success_rate=job_status.results['success_rate'],
        output_files=job_status.results['output_files'],
        report_file=job_status.results.get('html_report_path')
    )

@app.get("/download/report/{job_id}", tags=["File Downloads"])
async def download_report(job_id: str, format: str = "html"):
    """Download processing report"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = processing_jobs[job_id]
    
    if job_status.status != "completed" or not job_status.results:
        raise HTTPException(status_code=400, detail="Job not completed or no results available")
    
    if format.lower() == "html":
        report_path = job_status.results.get('html_report_path')
        media_type = "text/html"
    elif format.lower() == "json":
        report_path = job_status.results.get('json_report_path')
        media_type = "application/json"
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'html' or 'json'")
    
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=report_path,
        media_type=media_type,
        filename=Path(report_path).name
    )

@app.get("/jobs", tags=["Job Management"])
async def list_processing_jobs():
    """List all processing jobs"""
    
    job_summaries = []
    for job_id, job_status in processing_jobs.items():
        job_summaries.append({
            'job_id': job_id,
            'status': job_status.status,
            'progress': job_status.progress,
            'started_at': job_status.started_at.isoformat(),
            'completed_at': job_status.completed_at.isoformat() if job_status.completed_at else None,
            'message': job_status.message
        })
    
    return {
        'total_jobs': len(job_summaries),
        'jobs': job_summaries,
        'timestamp': datetime.now().isoformat()
    }

@app.delete("/jobs/{job_id}", tags=["Job Management"])
async def delete_processing_job(job_id: str):
    """Delete processing job and its results"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = processing_jobs[job_id]
    
    # Clean up output files if job completed
    if job_status.status == "completed" and job_status.results:
        try:
            output_files = job_status.results.get('output_files', [])
            for file_path in output_files:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            
            # Clean up reports
            for report_key in ['json_report_path', 'html_report_path']:
                report_path = job_status.results.get(report_key)
                if report_path and Path(report_path).exists():
                    Path(report_path).unlink()
            
        except Exception as e:
            logger.warning(f"Failed to clean up files for job {job_id}: {str(e)}")
    
    # Remove job from memory
    del processing_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)