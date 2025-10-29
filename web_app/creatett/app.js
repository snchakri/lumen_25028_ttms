// Fake Timetable Data
const timetableData = {
  metadata: {
    institution: "Demo Engineering College",
    department: "Computer Science & Engineering",
    semester: "5th Semester",
    academicYear: "2024-25",
    section: "A",
    generatedOn: "2025-10-28 13:38:00",
    totalCredits: 18,
    totalCourses: 5
  },
  schedule: [
    {
      day: "Monday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS305",
          courseName: "Web Technologies Lab",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "LAB-CS1",
          type: "lab",
          credits: 3
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS305",
          courseName: "Web Technologies Lab",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "LAB-CS1",
          type: "lab",
          credits: 3
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        }
      ]
    },
    {
      day: "Tuesday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS302",
          courseName: "Database Management Lab",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "LAB-CS2",
          type: "lab",
          credits: 3
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS302",
          courseName: "Database Management Lab",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "LAB-CS2",
          type: "lab",
          credits: 3
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS305",
          courseName: "Web Technologies",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        }
      ]
    },
    {
      day: "Wednesday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS305",
          courseName: "Web Technologies",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS303",
          courseName: "Operating Systems Lab",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "LAB-CS1",
          type: "lab",
          credits: 4
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS303",
          courseName: "Operating Systems Lab",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "LAB-CS1",
          type: "lab",
          credits: 4
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        }
      ]
    },
    {
      day: "Thursday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS304",
          courseName: "Computer Networks Lab",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "LAB-CS2",
          type: "lab",
          credits: 3
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS304",
          courseName: "Computer Networks Lab",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "LAB-CS2",
          type: "lab",
          credits: 3
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS305",
          courseName: "Web Technologies",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        }
      ]
    },
    {
      day: "Friday",
      classes: [
        {
          period: 1,
          time: "8:00-9:00",
          courseCode: "CS305",
          courseName: "Web Technologies",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: 2,
          time: "9:00-10:00",
          courseCode: "CS304",
          courseName: "Computer Networks",
          facultyName: "Prof. Amit Singh",
          facultyId: "FAC103",
          room: "B201",
          type: "theory",
          credits: 3
        },
        {
          period: 3,
          time: "10:00-11:00",
          courseCode: "CS301",
          courseName: "Data Structures",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        },
        {
          period: 4,
          time: "11:00-12:00",
          courseCode: "CS302",
          courseName: "Database Management",
          facultyName: "Dr. Priya Sharma",
          facultyId: "FAC102",
          room: "A102",
          type: "theory",
          credits: 3
        },
        {
          period: "Lunch",
          time: "12:00-1:00",
          courseName: "Lunch Break",
          type: "break"
        },
        {
          period: 5,
          time: "1:00-2:00",
          courseCode: "CS301",
          courseName: "Data Structures Lab",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "LAB-CS1",
          type: "lab",
          credits: 4
        },
        {
          period: 6,
          time: "2:00-3:00",
          courseCode: "CS301",
          courseName: "Data Structures Lab",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "LAB-CS1",
          type: "lab",
          credits: 4
        },
        {
          period: 7,
          time: "3:00-4:00",
          courseCode: "CS303",
          courseName: "Operating Systems",
          facultyName: "Dr. Rajesh Kumar",
          facultyId: "FAC101",
          room: "A101",
          type: "theory",
          credits: 4
        }
      ]
    }
  ],
  facultyLoad: [
    {
      facultyId: "FAC101",
      facultyName: "Dr. Rajesh Kumar",
      totalHours: 22,
      courses: ["CS301", "CS303"]
    },
    {
      facultyId: "FAC102",
      facultyName: "Dr. Priya Sharma",
      totalHours: 20,
      courses: ["CS302", "CS305"]
    },
    {
      facultyId: "FAC103",
      facultyName: "Prof. Amit Singh",
      totalHours: 14,
      courses: ["CS304"]
    }
  ],
  roomUtilization: [
    { room: "A101", utilization: "85%", totalHours: 34 },
    { room: "A102", utilization: "78%", totalHours: 31 },
    { room: "LAB-CS1", utilization: "92%", totalHours: 37 },
    { room: "LAB-CS2", utilization: "88%", totalHours: 35 },
    { room: "B201", utilization: "75%", totalHours: 30 }
  ]
};

// Application State
let currentData = timetableData;
let currentView = 'grid';
let selectedDay = 'all';
let searchQuery = '';
let currentPage = 'dashboard';
let uploadedFiles = {};
let selectedTimetable = null;
let generatedTimetables = [];

// DOM Elements
const modal = document.getElementById('timetableModal');
const viewTimetableBtn = document.getElementById('viewTimetableBtn');
const uploadCSVBtn = document.getElementById('uploadCSVBtn');
const closeModalBtn = document.getElementById('closeModalBtn');
const modalOverlay = document.querySelector('.modal-overlay');
const tabBtns = document.querySelectorAll('.tab-btn');
const viewContents = document.querySelectorAll('.view-content');
const dayFilter = document.getElementById('dayFilter');
const searchInput = document.getElementById('searchInput');
const exportBtn = document.getElementById('exportBtn');
const exportMenu = document.getElementById('exportMenu');
const darkModeToggle = document.getElementById('darkModeToggle');
const csvFileInput = document.getElementById('csvFileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadZone = document.getElementById('uploadZone');

// Initialize App
function init() {
  renderMetadata();
  renderGridView();
  renderStatistics();
  attachEventListeners();
}

// Render Metadata
function renderMetadata() {
  document.getElementById('institutionName').textContent = currentData.metadata.institution;
  document.getElementById('departmentName').textContent = currentData.metadata.department;
  document.getElementById('semesterName').textContent = currentData.metadata.semester;
}

// Render Grid View
function renderGridView() {
  const gridContainer = document.getElementById('timetableGrid');
  gridContainer.innerHTML = '';
  
  const days = selectedDay === 'all' 
    ? currentData.schedule.map(d => d.day) 
    : [selectedDay];
  
  const filteredSchedule = selectedDay === 'all' 
    ? currentData.schedule 
    : currentData.schedule.filter(d => d.day === selectedDay);
  
  // Get all unique time slots
  const allClasses = filteredSchedule.flatMap(d => d.classes);
  const timeSlots = [...new Set(allClasses.map(c => c.time))];
  
  // Create header row
  const cornerCell = document.createElement('div');
  cornerCell.className = 'grid-header';
  cornerCell.textContent = 'Time';
  gridContainer.appendChild(cornerCell);
  
  days.forEach(day => {
    const dayCell = document.createElement('div');
    dayCell.className = 'grid-header';
    dayCell.textContent = day;
    gridContainer.appendChild(dayCell);
  });
  
  // Create time rows
  timeSlots.forEach(time => {
    // Time cell
    const timeCell = document.createElement('div');
    timeCell.className = 'grid-cell time-cell';
    timeCell.textContent = time;
    gridContainer.appendChild(timeCell);
    
    // Class cells for each day
    filteredSchedule.forEach(daySchedule => {
      const classAtTime = daySchedule.classes.find(c => c.time === time);
      const classCell = document.createElement('div');
      classCell.className = 'grid-cell';
      
      if (classAtTime) {
        if (classAtTime.type === 'break') {
          classCell.classList.add('class-cell', 'break');
          classCell.innerHTML = `
            <div class="class-name">${classAtTime.courseName}</div>
          `;
        } else {
          classCell.classList.add('class-cell', classAtTime.type);
          classCell.innerHTML = `
            <div class="class-code">${classAtTime.courseCode}</div>
            <div class="class-name">${classAtTime.courseName}</div>
            <div class="class-faculty">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
              ${classAtTime.facultyName}
            </div>
            <div class="class-room">${classAtTime.room}</div>
          `;
          
          // Add hover tooltip
          classCell.addEventListener('mouseenter', (e) => showTooltip(e, classAtTime, daySchedule.day));
          classCell.addEventListener('mouseleave', hideTooltip);
        }
      }
      
      gridContainer.appendChild(classCell);
    });
  });
  
  // Adjust grid columns
  gridContainer.style.gridTemplateColumns = `100px repeat(${days.length}, 1fr)`;
}

// Render List View
function renderListView() {
  const listContainer = document.getElementById('listContainer');
  listContainer.innerHTML = '';
  
  const filteredSchedule = selectedDay === 'all' 
    ? currentData.schedule 
    : currentData.schedule.filter(d => d.day === selectedDay);
  
  filteredSchedule.forEach(daySchedule => {
    const daySection = document.createElement('div');
    daySection.className = 'day-section';
    
    const dayHeader = document.createElement('div');
    dayHeader.className = 'day-header';
    dayHeader.textContent = daySchedule.day;
    daySection.appendChild(dayHeader);
    
    const classList = document.createElement('div');
    classList.className = 'class-list';
    
    const filteredClasses = daySchedule.classes.filter(c => 
      c.type !== 'break' && 
      (searchQuery === '' || 
       c.courseName.toLowerCase().includes(searchQuery.toLowerCase()) ||
       c.courseCode?.toLowerCase().includes(searchQuery.toLowerCase()) ||
       c.facultyName?.toLowerCase().includes(searchQuery.toLowerCase()))
    );
    
    filteredClasses.forEach(classItem => {
      const item = document.createElement('div');
      item.className = 'class-item';
      item.innerHTML = `
        <div class="time-badge">${classItem.time}</div>
        <div class="class-details">
          <h4>${classItem.courseCode} - ${classItem.courseName}</h4>
          <div class="class-meta">
            <span>👤 ${classItem.facultyName}</span>
            <span>📍 ${classItem.room}</span>
            <span>📚 ${classItem.credits} Credits</span>
          </div>
        </div>
        <span class="type-badge ${classItem.type}">${classItem.type.toUpperCase()}</span>
      `;
      classList.appendChild(item);
    });
    
    daySection.appendChild(classList);
    listContainer.appendChild(daySection);
  });
}

// Render Faculty View
function renderFacultyView() {
  const facultyContainer = document.getElementById('facultyContainer');
  facultyContainer.innerHTML = '';
  
  currentData.facultyLoad.forEach(faculty => {
    const facultyCard = document.createElement('div');
    facultyCard.className = 'faculty-card';
    
    const facultyHeader = document.createElement('div');
    facultyHeader.className = 'faculty-header';
    facultyHeader.innerHTML = `
      <h3>${faculty.facultyName}</h3>
      <div class="faculty-info">
        <span>📊 ${faculty.totalHours} hours/week</span>
        <span>📚 ${faculty.courses.length} courses</span>
      </div>
    `;
    facultyCard.appendChild(facultyHeader);
    
    const facultySchedule = document.createElement('div');
    facultySchedule.className = 'faculty-schedule';
    
    // Get all classes for this faculty
    const facultyClasses = [];
    currentData.schedule.forEach(day => {
      day.classes.forEach(cls => {
        if (cls.facultyId === faculty.facultyId && cls.type !== 'break') {
          facultyClasses.push({ ...cls, day: day.day });
        }
      });
    });
    
    facultyClasses.forEach(cls => {
      const scheduleItem = document.createElement('div');
      scheduleItem.className = `schedule-item ${cls.type}`;
      scheduleItem.innerHTML = `
        <div class="schedule-time">${cls.day} • ${cls.time}</div>
        <div class="schedule-course">${cls.courseCode} - ${cls.courseName}</div>
        <div class="schedule-meta">Room: ${cls.room} • ${cls.type.toUpperCase()}</div>
      `;
      facultySchedule.appendChild(scheduleItem);
    });
    
    facultyCard.appendChild(facultySchedule);
    facultyContainer.appendChild(facultyCard);
  });
}

// Render Room View
function renderRoomView() {
  const roomContainer = document.getElementById('roomContainer');
  roomContainer.innerHTML = '';
  
  currentData.roomUtilization.forEach(roomData => {
    const roomCard = document.createElement('div');
    roomCard.className = 'room-card';
    
    const roomHeader = document.createElement('div');
    roomHeader.className = 'room-header';
    roomHeader.innerHTML = `
      <h3>${roomData.room}</h3>
      <div class="room-info">
        <span>📊 ${roomData.utilization} utilized</span>
        <span>⏱️ ${roomData.totalHours} hours/week</span>
      </div>
    `;
    roomCard.appendChild(roomHeader);
    
    const roomSchedule = document.createElement('div');
    roomSchedule.className = 'room-schedule';
    
    // Get all classes for this room
    const roomClasses = [];
    currentData.schedule.forEach(day => {
      day.classes.forEach(cls => {
        if (cls.room === roomData.room && cls.type !== 'break') {
          roomClasses.push({ ...cls, day: day.day });
        }
      });
    });
    
    roomClasses.forEach(cls => {
      const scheduleItem = document.createElement('div');
      scheduleItem.className = `schedule-item ${cls.type}`;
      scheduleItem.innerHTML = `
        <div class="schedule-time">${cls.day} • ${cls.time}</div>
        <div class="schedule-course">${cls.courseCode} - ${cls.courseName}</div>
        <div class="schedule-meta">Faculty: ${cls.facultyName} • ${cls.type.toUpperCase()}</div>
      `;
      roomSchedule.appendChild(scheduleItem);
    });
    
    roomCard.appendChild(roomSchedule);
    roomContainer.appendChild(roomCard);
  });
}

// Render Statistics
function renderStatistics() {
  let totalClasses = 0;
  let totalHours = 0;
  
  currentData.schedule.forEach(day => {
    day.classes.forEach(cls => {
      if (cls.type !== 'break') {
        totalClasses++;
        totalHours++;
      }
    });
  });
  
  document.getElementById('totalClasses').textContent = totalClasses;
  document.getElementById('totalHours').textContent = totalHours;
  document.getElementById('totalCredits').textContent = currentData.metadata.totalCredits;
}

// Show Tooltip
function showTooltip(event, classData, day) {
  const tooltip = document.getElementById('detailTooltip');
  const content = tooltip.querySelector('.tooltip-content');
  
  content.innerHTML = `
    <h4>${classData.courseCode} - ${classData.courseName}</h4>
    <p><strong>Day:</strong> ${day}</p>
    <p><strong>Time:</strong> ${classData.time}</p>
    <p><strong>Faculty:</strong> ${classData.facultyName}</p>
    <p><strong>Room:</strong> ${classData.room}</p>
    <p><strong>Type:</strong> ${classData.type.toUpperCase()}</p>
    <p><strong>Credits:</strong> ${classData.credits}</p>
  `;
  
  tooltip.style.display = 'block';
  tooltip.style.left = (event.pageX + 15) + 'px';
  tooltip.style.top = (event.pageY + 15) + 'px';
}

// Hide Tooltip
function hideTooltip() {
  const tooltip = document.getElementById('detailTooltip');
  tooltip.style.display = 'none';
}

// Switch View
function switchView(view) {
  currentView = view;
  
  // Update tabs
  tabBtns.forEach(btn => {
    if (btn.dataset.view === view) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  
  // Update views
  viewContents.forEach(content => {
    content.classList.remove('active');
  });
  
  const targetView = document.getElementById(`${view}View`);
  targetView.classList.add('active');
  
  // Render appropriate view
  switch(view) {
    case 'grid':
      renderGridView();
      break;
    case 'list':
      renderListView();
      break;
    case 'faculty':
      renderFacultyView();
      break;
    case 'room':
      renderRoomView();
      break;
  }
}

// Export Functions
function exportAsCSV() {
  let csv = 'Day,Time,Period,Course Code,Course Name,Faculty,Room,Type,Credits\n';
  
  currentData.schedule.forEach(day => {
    day.classes.forEach(cls => {
      if (cls.type !== 'break') {
        csv += `${day.day},${cls.time},${cls.period},${cls.courseCode},${cls.courseName},${cls.facultyName},${cls.room},${cls.type},${cls.credits}\n`;
      }
    });
  });
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `timetable_${currentData.metadata.semester}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportAsHTML() {
  let html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>Timetable - ${currentData.metadata.semester}</title>
      <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        h1 { color: #21808D; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #21808D; color: white; }
        .theory { background-color: rgba(59, 130, 246, 0.1); }
        .lab { background-color: rgba(34, 197, 94, 0.1); }
      </style>
    </head>
    <body>
      <h1>${currentData.metadata.institution}</h1>
      <h2>${currentData.metadata.department} - ${currentData.metadata.semester}</h2>
      <table>
        <thead>
          <tr>
            <th>Day</th>
            <th>Time</th>
            <th>Course Code</th>
            <th>Course Name</th>
            <th>Faculty</th>
            <th>Room</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
  `;
  
  currentData.schedule.forEach(day => {
    day.classes.forEach(cls => {
      if (cls.type !== 'break') {
        html += `
          <tr class="${cls.type}">
            <td>${day.day}</td>
            <td>${cls.time}</td>
            <td>${cls.courseCode}</td>
            <td>${cls.courseName}</td>
            <td>${cls.facultyName}</td>
            <td>${cls.room}</td>
            <td>${cls.type.toUpperCase()}</td>
          </tr>
        `;
      }
    });
  });
  
  html += `
        </tbody>
      </table>
    </body>
    </html>
  `;
  
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `timetable_${currentData.metadata.semester}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

function printTimetable() {
  window.print();
}

// Parse CSV
function parseCSV(csvText) {
  const lines = csvText.split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  
  const schedule = {};
  
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '') continue;
    
    const values = lines[i].split(',').map(v => v.trim());
    const day = values[0];
    
    if (!schedule[day]) {
      schedule[day] = { day, classes: [] };
    }
    
    schedule[day].classes.push({
      time: values[1],
      period: parseInt(values[2]) || values[2],
      courseCode: values[3],
      courseName: values[4],
      facultyName: values[5],
      room: values[6],
      type: values[7],
      credits: parseInt(values[8]) || 0
    });
  }
  
  return Object.values(schedule);
}

// Handle CSV Upload
function handleCSVUpload(file) {
  const reader = new FileReader();
  
  reader.onload = (e) => {
    try {
      const csvText = e.target.result;
      const parsedSchedule = parseCSV(csvText);
      
      currentData.schedule = parsedSchedule;
      uploadZone.style.display = 'none';
      switchView('grid');
      renderStatistics();
    } catch (error) {
      alert('Error parsing CSV file. Please check the format.');
    }
  };
  
  reader.readAsText(file);
}

// Event Listeners
function attachEventListeners() {
  // Open/Close Modal
  viewTimetableBtn.addEventListener('click', () => {
    modal.classList.add('active');
  });
  
  uploadCSVBtn.addEventListener('click', () => {
    modal.classList.add('active');
    uploadZone.style.display = 'flex';
    viewContents.forEach(v => v.classList.remove('active'));
  });
  
  closeModalBtn.addEventListener('click', () => {
    modal.classList.remove('active');
    uploadZone.style.display = 'none';
    document.getElementById('gridView').classList.add('active');
  });
  
  modalOverlay.addEventListener('click', () => {
    modal.classList.remove('active');
    uploadZone.style.display = 'none';
    document.getElementById('gridView').classList.add('active');
  });
  
  // View Tabs
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      switchView(btn.dataset.view);
    });
  });
  
  // Day Filter
  dayFilter.addEventListener('change', (e) => {
    selectedDay = e.target.value;
    switchView(currentView);
  });
  
  // Search
  searchInput.addEventListener('input', (e) => {
    searchQuery = e.target.value;
    if (currentView === 'list') {
      renderListView();
    }
  });
  
  // Export
  exportBtn.addEventListener('click', () => {
    exportMenu.classList.toggle('active');
  });
  
  document.addEventListener('click', (e) => {
    if (!exportBtn.contains(e.target) && !exportMenu.contains(e.target)) {
      exportMenu.classList.remove('active');
    }
  });
  
  document.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('click', (e) => {
      const exportType = e.target.dataset.export;
      
      switch(exportType) {
        case 'csv':
          exportAsCSV();
          break;
        case 'html':
          exportAsHTML();
          break;
        case 'print':
          printTimetable();
          break;
      }
      
      exportMenu.classList.remove('active');
    });
  });
  
  // Dark Mode Toggle
  darkModeToggle.addEventListener('click', () => {
    document.documentElement.setAttribute(
      'data-theme',
      document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
    );
  });
  
  // CSV Upload
  browseBtn.addEventListener('click', () => {
    csvFileInput.click();
  });
  
  csvFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleCSVUpload(e.target.files[0]);
    }
  });
  
  uploadZone.addEventListener('click', () => {
    csvFileInput.click();
  });
  
  // Prevent modal close when clicking inside
  document.querySelector('.modal-container').addEventListener('click', (e) => {
    e.stopPropagation();
  });
  
  // Page Navigation
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchPage(btn.dataset.page);
    });
  });
  
  // Create Page Event Listeners
  document.getElementById('fetchDbBtn').addEventListener('click', handleDatabaseFetch);
  document.getElementById('uploadFilesBtn').addEventListener('click', openUploadModal);
  document.getElementById('closeUploadModalBtn').addEventListener('click', closeUploadModal);
  document.getElementById('createTimetableBtn').addEventListener('click', handleFileSubmission);
  document.getElementById('publishBtn').addEventListener('click', publishTimetable);
  document.getElementById('discardAllBtn').addEventListener('click', discardAll);
  
  // Upload Modal Overlay
  document.querySelector('#uploadModal .modal-overlay').addEventListener('click', closeUploadModal);
  document.querySelector('#uploadModal .modal-container').addEventListener('click', (e) => {
    e.stopPropagation();
  });
  
  // Status Modal Overlay
  document.querySelector('#statusModal .modal-overlay').addEventListener('click', (e) => {
    // Don't allow closing during loading
    if (!document.querySelector('.status-icon.loading')) {
      closeStatusModal();
    }
  });
  
  // Delegate upload button clicks
  document.getElementById('uploadFilesList').addEventListener('click', (e) => {
    const uploadBtn = e.target.closest('.upload-btn');
    if (uploadBtn && !uploadBtn.classList.contains('uploaded')) {
      const index = uploadBtn.dataset.index;
      handleFileUpload(index);
    }
  });
}

// Create Timetable Data
const requiredFiles = [
  { name: "Students Master Data", description: "Complete list of all students", required: true },
  { name: "Courses Master Data", description: "All courses offered this semester", required: true },
  { name: "Faculty Master Data", description: "Faculty member details and qualifications", required: true },
  { name: "Rooms Master Data", description: "Available classrooms and labs", required: true },
  { name: "Time Slots Configuration", description: "Daily time slot structure", required: true },
  { name: "Course Prerequisites", description: "Course dependencies and requirements", required: false },
  { name: "Faculty Availability", description: "Faculty time preferences and constraints", required: true },
  { name: "Room Availability", description: "Room booking constraints", required: true },
  { name: "Student Enrollments", description: "Student course registrations", required: true },
  { name: "Course Sections", description: "Section divisions for each course", required: true },
  { name: "Lab Requirements", description: "Lab-specific constraints and equipment", required: false },
  { name: "Theory Requirements", description: "Theory class specific needs", required: false },
  { name: "Constraint Rules", description: "Custom scheduling constraints", required: true },
  { name: "Holiday Calendar", description: "Institutional holidays and breaks", required: false },
  { name: "Exam Schedule", description: "Examination dates and constraints", required: false },
  { name: "Faculty Load Limits", description: "Maximum teaching hours per faculty", required: true },
  { name: "Room Capacity Data", description: "Seating capacity for each room", required: true },
  { name: "Program Structure", description: "Degree program requirements", required: true },
  { name: "Batch Information", description: "Student batch and year details", required: true },
  { name: "Special Requirements", description: "Any additional scheduling needs", required: false }
];

// Page Navigation
function switchPage(page) {
  currentPage = page;
  
  const dashboard = document.querySelector('.dashboard');
  const createPage = document.getElementById('createPage');
  const navBtns = document.querySelectorAll('.nav-btn');
  
  navBtns.forEach(btn => {
    if (btn.dataset.page === page) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  
  if (page === 'dashboard') {
    dashboard.style.display = 'flex';
    createPage.style.display = 'none';
  } else if (page === 'create') {
    dashboard.style.display = 'none';
    createPage.style.display = 'block';
  }
}

// Upload Modal Functions
function openUploadModal() {
  const uploadModal = document.getElementById('uploadModal');
  const uploadFilesList = document.getElementById('uploadFilesList');
  
  uploadFilesList.innerHTML = '';
  uploadedFiles = {};
  
  requiredFiles.forEach((file, index) => {
    const fileRow = document.createElement('div');
    fileRow.className = 'upload-file-row';
    fileRow.dataset.index = index;
    
    fileRow.innerHTML = `
      <div class="file-info">
        <div class="file-name">${file.name}</div>
        <div class="file-description">${file.description}</div>
        ${file.required ? '<span class="file-required">* Required</span>' : ''}
      </div>
      <button class="upload-btn" data-index="${index}">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        Upload
      </button>
    `;
    
    uploadFilesList.appendChild(fileRow);
  });
  
  uploadModal.classList.add('active');
  updateUploadProgress();
}

function closeUploadModal() {
  const uploadModal = document.getElementById('uploadModal');
  uploadModal.classList.remove('active');
}

function updateUploadProgress() {
  const uploadedCount = Object.keys(uploadedFiles).length;
  const totalCount = requiredFiles.length;
  const percentage = (uploadedCount / totalCount) * 100;
  
  document.getElementById('uploadProgressFill').style.width = percentage + '%';
  document.getElementById('uploadProgressText').textContent = `${uploadedCount}/${totalCount} files uploaded`;
  
  const createBtn = document.getElementById('createTimetableBtn');
  const requiredCount = requiredFiles.filter(f => f.required).length;
  const requiredUploaded = Object.keys(uploadedFiles).filter(idx => requiredFiles[idx].required).length;
  
  if (requiredUploaded === requiredCount) {
    createBtn.disabled = false;
  } else {
    createBtn.disabled = true;
  }
}

function handleFileUpload(index) {
  const fileRow = document.querySelector(`[data-index="${index}"]`);
  const uploadBtn = fileRow.querySelector('.upload-btn');
  
  // Simulate file upload
  uploadBtn.innerHTML = '<span>Uploading...</span>';
  
  setTimeout(() => {
    uploadedFiles[index] = true;
    fileRow.classList.add('uploaded');
    uploadBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
      Uploaded
    `;
    uploadBtn.classList.add('uploaded');
    uploadBtn.disabled = true;
    updateUploadProgress();
  }, 500);
}

// Status Modal Functions
function showStatusModal(type, title, message, actions) {
  const statusModal = document.getElementById('statusModal');
  const statusContent = document.getElementById('statusModalContent');
  
  let iconHTML = '';
  if (type === 'success') {
    iconHTML = `
      <div class="status-icon success">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      </div>
    `;
  } else if (type === 'error') {
    iconHTML = `
      <div class="status-icon error">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="15" y1="9" x2="9" y2="15"></line>
          <line x1="9" y1="9" x2="15" y2="15"></line>
        </svg>
      </div>
    `;
  } else if (type === 'loading') {
    iconHTML = `
      <div class="status-icon loading">
        <div class="spinner"></div>
      </div>
    `;
  }
  
  let actionsHTML = '';
  if (actions && actions.length > 0) {
    actionsHTML = '<div class="status-actions">';
    actions.forEach(action => {
      actionsHTML += `<button class="btn ${action.type}" onclick="${action.onclick}">${action.label}</button>`;
    });
    actionsHTML += '</div>';
  }
  
  statusContent.innerHTML = `
    ${iconHTML}
    <h3 class="status-title">${title}</h3>
    <p class="status-message">${message}</p>
    ${actionsHTML}
  `;
  
  statusModal.classList.add('active');
}

function closeStatusModal() {
  const statusModal = document.getElementById('statusModal');
  statusModal.classList.remove('active');
}

// Database Fetch
function handleDatabaseFetch() {
  const dbCard = document.getElementById('dbFetchCard');
  dbCard.classList.add('loading');
  
  showStatusModal('loading', 'Fetching Data', 'Retrieving all required data from database...', []);
  
  setTimeout(() => {
    dbCard.classList.remove('loading');
    
    // Simulate random success/failure
    const success = Math.random() > 0.2;
    
    if (success) {
      closeStatusModal();
      setTimeout(() => {
        showStatusModal(
          'success',
          'Files Submitted Successfully!',
          'Your request has been queued for processing. The scheduling engine will process your data shortly.',
          [{ label: 'Done', type: 'btn--primary', onclick: 'handleSubmissionSuccess()' }]
        );
      }, 300);
    } else {
      closeStatusModal();
      setTimeout(() => {
        showStatusModal(
          'error',
          'Submission Failed',
          'Failed to connect to database. Please check connection settings.',
          [{ label: 'Retry', type: 'btn--primary', onclick: 'closeStatusModal()' }]
        );
      }, 300);
    }
  }, 2000);
}

// File Upload Submission
function handleFileSubmission() {
  closeUploadModal();
  
  showStatusModal('loading', 'Uploading Files', 'Uploading files to database and queuing for processing...', []);
  
  setTimeout(() => {
    const success = Math.random() > 0.2;
    
    closeStatusModal();
    setTimeout(() => {
      if (success) {
        showStatusModal(
          'success',
          'Files Submitted Successfully!',
          'All files uploaded successfully! Your request has been queued for processing.',
          [{ label: 'Done', type: 'btn--primary', onclick: 'handleSubmissionSuccess()' }]
        );
      } else {
        showStatusModal(
          'error',
          'Upload Failed',
          'Upload failed: Missing required file \'Students Master Data\'',
          [{ label: 'Retry', type: 'btn--primary', onclick: 'closeStatusModal()' }]
        );
      }
    }, 300);
  }, 2000);
}

// Handle Submission Success
function handleSubmissionSuccess() {
  closeStatusModal();
  
  // Simulate engine processing
  setTimeout(() => {
    const success = Math.random() > 0.3;
    
    if (success) {
      showStatusModal(
        'success',
        'Generation Successful!',
        'Timetable generation completed successfully! 5 variations generated.',
        [{ label: 'Done', type: 'btn--primary', onclick: 'handleEngineSuccess()' }]
      );
    } else {
      showStatusModal(
        'error',
        'Generation Failed',
        'Timetable generation failed: Constraint conflicts detected in faculty availability.',
        [
          { label: 'Download Error Report', type: 'btn--secondary', onclick: 'downloadErrorReport()' },
          { label: 'Close', type: 'btn--primary', onclick: 'closeStatusModal()' }
        ]
      );
    }
  }, 5000);
}

// Handle Engine Success
function handleEngineSuccess() {
  closeStatusModal();
  
  // Generate 5 timetables
  generatedTimetables = [
    { id: 'tt-1', name: 'TimeTable-1', qualityScore: 95, conflicts: 0, generatedAt: new Date().toISOString() },
    { id: 'tt-2', name: 'TimeTable-2', qualityScore: 92, conflicts: 2, generatedAt: new Date().toISOString() },
    { id: 'tt-3', name: 'TimeTable-3', qualityScore: 88, conflicts: 5, generatedAt: new Date().toISOString() },
    { id: 'tt-4', name: 'TimeTable-4', qualityScore: 85, conflicts: 8, generatedAt: new Date().toISOString() },
    { id: 'tt-5', name: 'TimeTable-5', qualityScore: 82, conflicts: 12, generatedAt: new Date().toISOString() }
  ];
  
  renderGeneratedTimetables();
}

// Render Generated Timetables
function renderGeneratedTimetables() {
  const generatedSection = document.getElementById('generatedSection');
  const container = document.getElementById('timetablesContainer');
  const actionButtons = document.getElementById('actionButtons');
  
  if (generatedTimetables.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
          <line x1="16" y1="2" x2="16" y2="6"></line>
          <line x1="8" y1="2" x2="8" y2="6"></line>
          <line x1="3" y1="10" x2="21" y2="10"></line>
        </svg>
        <p class="empty-message">No timetables generated yet. Create a new timetable above to get started.</p>
      </div>
    `;
    actionButtons.style.display = 'none';
  } else {
    container.innerHTML = '';
    
    generatedTimetables.forEach(tt => {
      const item = document.createElement('div');
      item.className = 'timetable-item';
      item.dataset.id = tt.id;
      
      item.innerHTML = `
        <div class="timetable-radio"></div>
        <div class="timetable-info">
          <div class="timetable-name">${tt.name}</div>
          <div class="timetable-meta">
            <span>Quality: ${tt.qualityScore}%</span>
            <span>Conflicts: ${tt.conflicts}</span>
            <span>Generated: ${new Date(tt.generatedAt).toLocaleString()}</span>
          </div>
        </div>
        <div class="timetable-actions">
          <button class="btn btn--secondary btn-small" onclick="viewTimetablePreview('${tt.id}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
              <circle cx="12" cy="12" r="3"></circle>
            </svg>
            View
          </button>
          <button class="btn btn--secondary btn-small" onclick="downloadTimetable('${tt.id}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Download
          </button>
        </div>
      `;
      
      item.addEventListener('click', (e) => {
        if (!e.target.closest('button')) {
          selectTimetable(tt.id);
        }
      });
      
      container.appendChild(item);
    });
    
    actionButtons.style.display = 'flex';
  }
  
  generatedSection.style.display = 'block';
}

// Select Timetable
function selectTimetable(id) {
  selectedTimetable = id;
  
  document.querySelectorAll('.timetable-item').forEach(item => {
    if (item.dataset.id === id) {
      item.classList.add('selected');
    } else {
      item.classList.remove('selected');
    }
  });
  
  document.getElementById('publishBtn').disabled = false;
}

// View Timetable Preview
function viewTimetablePreview(id) {
  modal.classList.add('active');
}

// Download Timetable
function downloadTimetable(id) {
  const tt = generatedTimetables.find(t => t.id === id);
  alert(`Downloading ${tt.name}...`);
}

// Download Error Report
function downloadErrorReport() {
  alert('Downloading error report...');
  closeStatusModal();
}

// Publish Timetable
function publishTimetable() {
  if (!selectedTimetable) return;
  
  const tt = generatedTimetables.find(t => t.id === selectedTimetable);
  showStatusModal(
    'success',
    'Timetable Published!',
    `${tt.name} has been published successfully and is now live.`,
    [{ label: 'Done', type: 'btn--primary', onclick: 'resetCreatePage()' }]
  );
}

// Discard All
function discardAll() {
  if (confirm('Are you sure you want to discard all generated timetables?')) {
    generatedTimetables = [];
    selectedTimetable = null;
    renderGeneratedTimetables();
    document.getElementById('generatedSection').style.display = 'none';
  }
}

// Reset Create Page
function resetCreatePage() {
  closeStatusModal();
  generatedTimetables = [];
  selectedTimetable = null;
  uploadedFiles = {};
  document.getElementById('generatedSection').style.display = 'none';
}

// Initialize on load
init();

// Make functions globally accessible for onclick handlers
window.closeStatusModal = closeStatusModal;
window.handleSubmissionSuccess = handleSubmissionSuccess;
window.handleEngineSuccess = handleEngineSuccess;
window.downloadErrorReport = downloadErrorReport;
window.viewTimetablePreview = viewTimetablePreview;
window.downloadTimetable = downloadTimetable;
window.resetCreatePage = resetCreatePage;