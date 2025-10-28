"""
Human-Readable Timetable Formatter for Stage-7 Validation Output

This module provides a clean, human-friendly view of validated timetables,
organized by department, day, and time slot in ascending order.

Author: GitHub Copilot, LUMEN Team [TEAM-ID: 93912]
Date: October 18, 2025
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json


class HumanReadableFormatter:
    """
    Formats validated timetables into human-readable views.
    
    Features:
    - Department-wise organization
    - Day-wise grouping (Monday → Friday)
    - Time slots in ascending order
    - Clean column subset (removes internal IDs)
    - Optional HTML/Markdown export
    """
    
    def __init__(self):
        """Initialize the formatter."""
        self.day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
    def format_timetable(
        self,
        schedule_df: pd.DataFrame,
        stage3_data: Dict[str, pd.DataFrame],
        output_format: str = "text"
    ) -> str:
        """
        Format timetable for human readability.
        
        Args:
            schedule_df: Validated schedule DataFrame
            stage3_data: Stage-3 data dictionary (courses, faculty, rooms, etc.)
            output_format: Output format ('text', 'markdown', 'html', 'json')
            
        Returns:
            Formatted timetable as string
        """
        # Enrich schedule with human-readable names
        enriched = self._enrich_schedule(schedule_df, stage3_data)
        
        # Sort by department, day, time
        sorted_schedule = self._sort_schedule(enriched)
        
        # Format based on output type
        if output_format == "text":
            return self._format_text(sorted_schedule)
        elif output_format == "markdown":
            return self._format_markdown(sorted_schedule)
        elif output_format == "html":
            return self._format_html(sorted_schedule)
        elif output_format == "json":
            return self._format_json(sorted_schedule)
        elif output_format == "csv":
            return self._format_csv(sorted_schedule)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def _enrich_schedule(
        self,
        schedule_df: pd.DataFrame,
        stage3_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Enrich schedule with human-readable names from Stage-3 data.
        
        Args:
            schedule_df: Raw schedule DataFrame
            stage3_data: Stage-3 data dictionary
            
        Returns:
            Enriched DataFrame with names
        """
        enriched = schedule_df.copy()
        
        # Add course names
        if 'courses' in stage3_data:
            courses = stage3_data['courses']
            # Handle both 'course_name' and 'name' columns
            name_col = 'course_name' if 'course_name' in courses.columns else 'name' if 'name' in courses.columns else None
            code_col = 'course_code' if 'course_code' in courses.columns else 'code' if 'code' in courses.columns else None
            
            if name_col:
                merge_cols = ['course_id', name_col, 'department_id']
                rename_dict = {name_col: 'Course'}
                
                if code_col and code_col in courses.columns:
                    merge_cols.append(code_col)
                    rename_dict[code_col] = 'Code'
                
                enriched = enriched.merge(
                    courses[merge_cols].rename(columns=rename_dict),
                    on='course_id',
                    how='left'
                )
        
        # Add faculty names
        if 'faculty' in stage3_data:
            faculty = stage3_data['faculty']
            # Handle both 'faculty_name' and 'name' columns
            name_col = 'faculty_name' if 'faculty_name' in faculty.columns else 'name' if 'name' in faculty.columns else None
            
            if name_col:
                enriched = enriched.merge(
                    faculty[['faculty_id', name_col]].rename(
                        columns={name_col: 'Instructor'}
                    ),
                    on='faculty_id',
                    how='left'
                )
        
        # Add room names
        if 'rooms' in stage3_data:
            rooms = stage3_data['rooms']
            # Handle both 'room_name' and 'name' columns
            name_col = 'room_name' if 'room_name' in rooms.columns else 'name' if 'name' in rooms.columns else None
            
            if name_col:
                merge_cols = ['room_id', name_col]
                rename_dict = {name_col: 'Room'}
                
                if 'building' in rooms.columns:
                    merge_cols.append('building')
                    rename_dict['building'] = 'Building'
                
                enriched = enriched.merge(
                    rooms[merge_cols].rename(columns=rename_dict),
                    on='room_id',
                    how='left'
                )
        
        # Add batch/section info
        batches_df = None
        if 'batches' in stage3_data:
            batches_df = stage3_data['batches']
        elif 'student_batches' in stage3_data:  # Fallback naming
            batches_df = stage3_data['student_batches']
        
        if batches_df is not None:
            # Handle both 'batch_name' and 'name' columns
            name_col = 'batch_name' if 'batch_name' in batches_df.columns else 'name' if 'name' in batches_df.columns else None
            
            if name_col:
                merge_cols = ['batch_id', name_col]
                rename_dict = {name_col: 'Section'}
                
                if 'semester' in batches_df.columns:
                    merge_cols.append('semester')
                    rename_dict['semester'] = 'Semester'
                if 'year' in batches_df.columns:
                    merge_cols.append('year')
                    rename_dict['year'] = 'Year'
                
                enriched = enriched.merge(
                    batches_df[merge_cols].rename(columns=rename_dict),
                    on='batch_id',
                    how='left'
                )
        
        # Add department names
        if 'departments' in stage3_data and 'department_id' in enriched.columns:
            departments = stage3_data['departments']
            # Handle both 'department_name' and 'name' columns
            name_col = 'department_name' if 'department_name' in departments.columns else 'name' if 'name' in departments.columns else None
            
            if name_col:
                enriched = enriched.merge(
                    departments[['department_id', name_col]].rename(
                        columns={name_col: 'Department'}
                    ),
                    on='department_id',
                    how='left'
                )
        
        # Add timeslot details
        if 'timeslots' in stage3_data:
            timeslots = stage3_data['timeslots']
        elif 'time_slots' in stage3_data:  # Fallback naming
            timeslots = stage3_data['time_slots']
        else:
            timeslots = None
        
        if timeslots is not None:
            if 'day' in timeslots.columns and 'start_time' in timeslots.columns:
                enriched = enriched.merge(
                    timeslots[['timeslot_id', 'day', 'start_time', 'end_time']].rename(
                        columns={'day': 'Day', 'start_time': 'Start', 'end_time': 'End'}
                    ),
                    on='timeslot_id',
                    how='left'
                )
        
        # Fallback: use schedule's own day/time if available
        if 'Day' not in enriched.columns and 'day' in enriched.columns:
            enriched['Day'] = enriched['day']
        if 'Start' not in enriched.columns and 'time' in enriched.columns:
            enriched['Start'] = enriched['time']
        if 'duration' in enriched.columns:
            enriched['Duration'] = enriched['duration'].astype(str) + ' min'
        
        return enriched
    
    def _sort_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort schedule by department, day (Monday→Friday), and time.
        
        Args:
            df: Enriched schedule DataFrame
            
        Returns:
            Sorted DataFrame
        """
        sorted_df = df.copy()
        
        # Add day order for sorting
        if 'Day' in sorted_df.columns:
            sorted_df['_day_order'] = sorted_df['Day'].map(
                {day: i for i, day in enumerate(self.day_order)}
            ).fillna(999)  # Unknown days go to end
        else:
            sorted_df['_day_order'] = 0
        
        # Sort by department, day, start time
        sort_cols = []
        if 'Department' in sorted_df.columns:
            sort_cols.append('Department')
        sort_cols.append('_day_order')
        if 'Start' in sorted_df.columns:
            sort_cols.append('Start')
        
        if sort_cols:
            sorted_df = sorted_df.sort_values(sort_cols)
        
        # Drop temporary sorting column
        if '_day_order' in sorted_df.columns:
            sorted_df = sorted_df.drop(columns=['_day_order'])
        
        return sorted_df
    
    def _format_text(self, df: pd.DataFrame) -> str:
        """Format timetable as plain text."""
        output = []
        output.append("=" * 100)
        output.append("VALIDATED TIMETABLE - HUMAN-READABLE VIEW")
        output.append("=" * 100)
        output.append("")
        
        # Select human-readable columns only
        display_cols = []
        for col in ['Department', 'Day', 'Start', 'End', 'Duration', 'Code', 'Course', 
                    'Instructor', 'Section', 'Room', 'Building', 'Semester', 'Year']:
            if col in df.columns:
                display_cols.append(col)
        
        if not display_cols:
            output.append("No displayable columns found.")
            return "\n".join(output)
        
        # Group by department if available
        if 'Department' in df.columns:
            for dept in df['Department'].dropna().unique():
                dept_df = df[df['Department'] == dept]
                output.append(f"\n{'─' * 100}")
                output.append(f"DEPARTMENT: {dept}")
                output.append(f"{'─' * 100}")
                
                # Group by day within department
                if 'Day' in dept_df.columns:
                    for day in self.day_order:
                        day_df = dept_df[dept_df['Day'] == day]
                        if not day_df.empty:
                            output.append(f"\n  {day}:")
                            output.append(f"  {'-' * 95}")
                            
                            # Format each assignment
                            for _, row in day_df.iterrows():
                                line_parts = []
                                if 'Start' in row and pd.notna(row['Start']):
                                    line_parts.append(f"{row['Start']}")
                                if 'End' in row and pd.notna(row['End']):
                                    line_parts.append(f"- {row['End']}")
                                if 'Code' in row and pd.notna(row['Code']):
                                    line_parts.append(f"| {row['Code']}")
                                if 'Course' in row and pd.notna(row['Course']):
                                    line_parts.append(f"- {row['Course']}")
                                if 'Instructor' in row and pd.notna(row['Instructor']):
                                    line_parts.append(f"| Prof. {row['Instructor']}")
                                if 'Section' in row and pd.notna(row['Section']):
                                    line_parts.append(f"| {row['Section']}")
                                if 'Room' in row and pd.notna(row['Room']):
                                    line_parts.append(f"| {row['Room']}")
                                if 'Building' in row and pd.notna(row['Building']):
                                    line_parts.append(f"({row['Building']})")
                                
                                output.append(f"    {' '.join(line_parts)}")
                else:
                    # No day grouping, just list all
                    for _, row in dept_df.iterrows():
                        line_parts = []
                        if 'Start' in row and pd.notna(row['Start']):
                            line_parts.append(f"{row['Start']}")
                        if 'Code' in row and pd.notna(row['Code']):
                            line_parts.append(f"| {row['Code']}")
                        if 'Course' in row and pd.notna(row['Course']):
                            line_parts.append(f"- {row['Course']}")
                        if 'Instructor' in row and pd.notna(row['Instructor']):
                            line_parts.append(f"| Prof. {row['Instructor']}")
                        if 'Room' in row and pd.notna(row['Room']):
                            line_parts.append(f"| {row['Room']}")
                        
                        output.append(f"  {' '.join(line_parts)}")
        else:
            # No department grouping
            output.append("\nALL CLASSES:")
            output.append("─" * 100)
            
            if 'Day' in df.columns:
                for day in self.day_order:
                    day_df = df[df['Day'] == day]
                    if not day_df.empty:
                        output.append(f"\n{day}:")
                        output.append("-" * 95)
                        
                        for _, row in day_df.iterrows():
                            line_parts = []
                            if 'Start' in row and pd.notna(row['Start']):
                                line_parts.append(f"{row['Start']}")
                            if 'End' in row and pd.notna(row['End']):
                                line_parts.append(f"- {row['End']}")
                            if 'Code' in row and pd.notna(row['Code']):
                                line_parts.append(f"| {row['Code']}")
                            if 'Course' in row and pd.notna(row['Course']):
                                line_parts.append(f"- {row['Course']}")
                            if 'Instructor' in row and pd.notna(row['Instructor']):
                                line_parts.append(f"| Prof. {row['Instructor']}")
                            if 'Room' in row and pd.notna(row['Room']):
                                line_parts.append(f"| {row['Room']}")
                            
                            output.append(f"  {' '.join(line_parts)}")
            else:
                # Just list all rows
                for _, row in df.iterrows():
                    line_parts = []
                    for col in display_cols:
                        if col in row and pd.notna(row[col]):
                            line_parts.append(f"{col}: {row[col]}")
                    output.append(f"  {' | '.join(line_parts)}")
        
        output.append("\n" + "=" * 100)
        output.append(f"Total Classes: {len(df)}")
        output.append("=" * 100)
        
        return "\n".join(output)
    
    def _format_markdown(self, df: pd.DataFrame) -> str:
        """Format timetable as Markdown."""
        output = []
        output.append("# Validated Timetable - Human-Readable View\n")
        
        # Select columns
        display_cols = []
        for col in ['Day', 'Start', 'End', 'Code', 'Course', 'Instructor', 'Section', 'Room']:
            if col in df.columns:
                display_cols.append(col)
        
        if not display_cols:
            output.append("No displayable columns found.\n")
            return "\n".join(output)
        
        # Group by department
        if 'Department' in df.columns:
            for dept in df['Department'].dropna().unique():
                dept_df = df[df['Department'] == dept]
                output.append(f"\n## {dept}\n")
                
                # Group by day
                if 'Day' in dept_df.columns:
                    for day in self.day_order:
                        day_df = dept_df[dept_df['Day'] == day]
                        if not day_df.empty:
                            output.append(f"### {day}\n")
                            
                            # Create markdown table
                            output.append("| " + " | ".join(display_cols) + " |")
                            output.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
                            
                            for _, row in day_df.iterrows():
                                row_data = [str(row.get(col, '')) if pd.notna(row.get(col)) else '' for col in display_cols]
                                output.append("| " + " | ".join(row_data) + " |")
                            
                            output.append("")
                else:
                    # Table without day grouping
                    output.append("| " + " | ".join(display_cols) + " |")
                    output.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
                    
                    for _, row in dept_df.iterrows():
                        row_data = [str(row.get(col, '')) if pd.notna(row.get(col)) else '' for col in display_cols]
                        output.append("| " + " | ".join(row_data) + " |")
                    
                    output.append("")
        else:
            # Single table
            output.append("\n## All Classes\n")
            output.append("| " + " | ".join(display_cols) + " |")
            output.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
            
            for _, row in df.iterrows():
                row_data = [str(row.get(col, '')) if pd.notna(row.get(col)) else '' for col in display_cols]
                output.append("| " + " | ".join(row_data) + " |")
            
            output.append("")
        
        output.append(f"\n**Total Classes:** {len(df)}\n")
        
        return "\n".join(output)
    
    def _format_html(self, df: pd.DataFrame) -> str:
        """Format timetable as HTML."""
        output = []
        output.append("<!DOCTYPE html>")
        output.append("<html><head>")
        output.append("<title>Validated Timetable</title>")
        output.append("<style>")
        output.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        output.append("h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }")
        output.append("h2 { color: #007bff; margin-top: 30px; }")
        output.append("h3 { color: #555; margin-top: 20px; }")
        output.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
        output.append("th { background-color: #007bff; color: white; padding: 10px; text-align: left; }")
        output.append("td { border: 1px solid #ddd; padding: 8px; }")
        output.append("tr:nth-child(even) { background-color: #f9f9f9; }")
        output.append("tr:hover { background-color: #f1f1f1; }")
        output.append(".summary { margin-top: 20px; font-weight: bold; color: #555; }")
        output.append("</style>")
        output.append("</head><body>")
        output.append("<h1>Validated Timetable - Human-Readable View</h1>")
        
        # Select columns
        display_cols = []
        for col in ['Day', 'Start', 'End', 'Code', 'Course', 'Instructor', 'Section', 'Room', 'Building']:
            if col in df.columns:
                display_cols.append(col)
        
        if not display_cols:
            output.append("<p>No displayable columns found.</p>")
            output.append("</body></html>")
            return "\n".join(output)
        
        # Group by department
        if 'Department' in df.columns:
            for dept in df['Department'].dropna().unique():
                dept_df = df[df['Department'] == dept]
                output.append(f"<h2>{dept}</h2>")
                
                # Group by day
                if 'Day' in dept_df.columns:
                    for day in self.day_order:
                        day_df = dept_df[dept_df['Day'] == day]
                        if not day_df.empty:
                            output.append(f"<h3>{day}</h3>")
                            output.append("<table>")
                            output.append("<tr>" + "".join([f"<th>{col}</th>" for col in display_cols]) + "</tr>")
                            
                            for _, row in day_df.iterrows():
                                output.append("<tr>")
                                for col in display_cols:
                                    val = row.get(col, '')
                                    output.append(f"<td>{val if pd.notna(val) else ''}</td>")
                                output.append("</tr>")
                            
                            output.append("</table>")
                else:
                    output.append("<table>")
                    output.append("<tr>" + "".join([f"<th>{col}</th>" for col in display_cols]) + "</tr>")
                    
                    for _, row in dept_df.iterrows():
                        output.append("<tr>")
                        for col in display_cols:
                            val = row.get(col, '')
                            output.append(f"<td>{val if pd.notna(val) else ''}</td>")
                        output.append("</tr>")
                    
                    output.append("</table>")
        else:
            output.append("<h2>All Classes</h2>")
            output.append("<table>")
            output.append("<tr>" + "".join([f"<th>{col}</th>" for col in display_cols]) + "</tr>")
            
            for _, row in df.iterrows():
                output.append("<tr>")
                for col in display_cols:
                    val = row.get(col, '')
                    output.append(f"<td>{val if pd.notna(val) else ''}</td>")
                output.append("</tr>")
            
            output.append("</table>")
        
        output.append(f"<p class='summary'>Total Classes: {len(df)}</p>")
        output.append("</body></html>")
        
        return "\n".join(output)
    
    def _format_json(self, df: pd.DataFrame) -> str:
        """Format timetable as structured JSON."""
        # Select human-readable columns
        display_cols = []
        for col in ['Department', 'Day', 'Start', 'End', 'Duration', 'Code', 'Course', 
                    'Instructor', 'Section', 'Room', 'Building', 'Semester', 'Year']:
            if col in df.columns:
                display_cols.append(col)
        
        if not display_cols:
            return json.dumps({"error": "No displayable columns found"}, indent=2)
        
        # Build structured output
        result = {
            "timetable": [],
            "summary": {
                "total_classes": len(df),
                "departments": df['Department'].nunique() if 'Department' in df.columns else 0,
                "days": df['Day'].nunique() if 'Day' in df.columns else 0
            }
        }
        
        # Group by department
        if 'Department' in df.columns:
            for dept in df['Department'].dropna().unique():
                dept_df = df[df['Department'] == dept]
                dept_data = {
                    "department": dept,
                    "classes": []
                }
                
                for _, row in dept_df.iterrows():
                    class_data = {}
                    for col in display_cols:
                        if col in row and pd.notna(row[col]):
                            class_data[col] = str(row[col])
                    dept_data["classes"].append(class_data)
                
                result["timetable"].append(dept_data)
        else:
            # No department grouping
            for _, row in df.iterrows():
                class_data = {}
                for col in display_cols:
                    if col in row and pd.notna(row[col]):
                        class_data[col] = str(row[col])
                result["timetable"].append(class_data)
        
        return json.dumps(result, indent=2)
    
    def _format_csv(self, df: pd.DataFrame) -> str:
        """Format timetable as clean CSV."""
        # Select human-readable columns only
        display_cols = []
        for col in ['Department', 'Day', 'Start', 'End', 'Duration', 'Code', 'Course', 
                    'Instructor', 'Section', 'Room', 'Building', 'Semester', 'Year']:
            if col in df.columns:
                display_cols.append(col)
        
        if not display_cols:
            return "No displayable columns found."
        
        # Create clean CSV output
        import io
        output = io.StringIO()
        df[display_cols].to_csv(output, index=False)
        return output.getvalue()
    
    def save_formatted_timetable(
        self,
        schedule_df: pd.DataFrame,
        stage3_data: Dict[str, pd.DataFrame],
        output_path: Path,
        output_format: str = "text"
    ) -> None:
        """
        Save formatted timetable to file.
        
        Args:
            schedule_df: Validated schedule DataFrame
            stage3_data: Stage-3 data dictionary
            output_path: Output file path
            output_format: Output format ('text', 'markdown', 'html', 'json')
        """
        formatted = self.format_timetable(schedule_df, stage3_data, output_format)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted)
