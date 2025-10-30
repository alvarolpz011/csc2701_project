import base64
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
# Load environment variables from .env file
load_dotenv()

TEMP_CHUNKS = \
[
"""
MScAC
MScAC
MScAC
MScAC
MScAC
STUDENT HANDBOOK 2025–2026
MScAC – STUDENT HANDBOOK 2025/26
Congratulations on your acceptance to the Master of Science in Applied Computing (MScAC) program!
The MScAC Student Handbook describes degree requirements, financial support, and other matters of
interest to MScAC students. The handbook is revised annually. Students will be notified by e-mail of
significant changes and upcoming deadlines. Please visit the MScAC website regularly at
mscac.utoronto.ca
DEPARTMENT BUILDINGS
The Department of Computer Science is located in four buildings on the downtown (St. George) campus
of the University of Toronto:
• 700 University (9th Floor, 700 University Avenue) [MScAC offices and student space are here]
• Bahen Centre for Information Technology (40 St. George Street)
• D.L. Pratt Building (6 King’s College Road)
• Sandford Fleming Building (10 King’s College Road)
""",

"""
IMPORTANT CONTACTS
ROLE NAME Academic Director,
Professional Programs Arvind Gupta Associate Chair, Graduate
Students Faith Ellen Associate Chair, Graduate
Operations Angela Demke Brown Associate Director, Academic Paul Gries Concentration Lead, Applied
Mathematics Mary Pugh Concentration Lead, Artificial
Intelligence in Healthcare Anna Goldenberg Concentration Lead,
Data Science Meredith Franklin Concentration Lead, Data
Science for Biology Qian Lin Concentration Lead, Quantum
Computing TBA Associate Director,
Partnerships Daniel Giovannini Development Officer Research and Business
Maurizio Ficocelli Development Officer Research and Business
Sulav Sharma Development Officer Research and Business
Daniele Chirico Development Officer Research and Business
Murtuza Rajkotwala Development Officer Research and Business
Siphelele Danisa Associate Director, MScAC
Administration Claire Mosses Interim Program Manager,
""",

"""
IMPORTANT DATES 2025/26
Fall 2025
First draft of study plans due June 13
Pre-enrolment courses available to view July 23
Data Science Bootcamp (Data Science Concentration students only) July 7 – July 30
Registration opens July 14
Enrolment in MAT courses begins TBC
Enrolment in CS Fall and Winter courses begins July 28
Enrolment in STA courses begins
(Data Science concentration students) TBC
MScAC Refresher Training (Linear algebra & probability) July 28 – August 1
Enrolment in PHY courses begins TBC
Enrolment in ECE courses begins TBC
Enrolment in CSB/MMG courses begins TBC
Enrolment in STA courses beings
(Other concentration students) TBC
Recommended tuition fee payment deadline for fees applicable to the Fall
semester. August 22
Orientation 2025 & Communication for Computer Scientists starts. MScAC
Refresher Training will be delivered during this week. Week beginning August 25
Clearing admission conditions August 31
Fall graduate courses in CS begin* September 2
Registration ends. Payment deadline for any unpaid Fall semester tuition
and fees. September 12
Final date to add Fall courses September 17
Final date to drop Fall courses without academic penalty October 27
MScAC Student Personal Time Off October 27 – October 31
Applied Research in Action (ARIA) November 13
Payment deadline for any unpaid Winter semester tuition and fees November 30
Fall term ends December 23
University closed for winter break December 24, 2025 –
January 4, 2026
Winter 2026
University re-opens January 5
Winter graduate courses in CS begin* January 5
Fall 2025 course grades available January 14
Final date to add Winter courses January 19
MScAC Internship Expo January 19 – January 23
MScAC Student Personal Time Off February 16 – 20
Final date to drop Winter courses without academic penalty February 27
Winter term classes end April
""",

"""
FEES AND FINANCES
The MScAC is a stand-alone program that is not funded through the Department of Computer Science
operating budget. Students in the program do not generally have an option to defer their fees*. You are
expected to pay the minimum amount to register by September 12, 2025, to avoid cancellation of your
“invited” registration status.
Domestic students may be eligible for government loans such as OSAP, the Ontario Student Assistance
Program. See: ontario.ca/page/osap-ontario-student-assistance-program
You are eligible to apply for Teaching Assistantship positions. These will be posted in late June/early
July, and all students in the graduate programs are invited to apply at that time. Please note you will
need to apply for a TA position to be made an offer. Without an application, positions will not be offered.
You will be notified about the course(s) for which you were selected as a Teaching Assistant before or
during the first full week of September.
Students in financial difficulty may wish to consult a Financial Advisor at the School of Graduate Studies,
63 St. George Street. An advisor can help with budgeting and may have knowledge of various bursaries,
grants, loans or other financial aid to help a student experiencing financial hardship.
See: sgs.utoronto.ca/awards-funding/financial-aid-advising
*Students in receipt of OSAP, CSL, US student loans, or any major awards such as the Vector Scholarship in AI that
cover the Minimum Required Payment may be able to defer their fees.
""",

"""
COURSE INFORMATION
Course Overview
The MScAC program is a 16-month applied research program designed to educate the next generation
of world-class innovators. Students enrol in advanced graduate courses according to the concentration
requirements. They also complete an eight-month applied research internship, usually paid, based at
an industry partner.
Typical program schedule for MScAC students
Year 1
Semester 1:
September – December
Year 1
Semester 2:
January – April
Year 1
Semester 3:
May – August
Year 2
Semester 4:
September – December
CSC2701H CSC2701H CSC2702H CSC2702H
Two approved
graduate courses
Two approved
graduate courses
Applied research
internship
Applied research
Internship
Resume preparation
Internship search begins
Industry partner nights
MScAC Internship
Expo & interviews
Applied Research in
Action (ARIA) Showcase
+ final report submission
""",

"""
COURSE REQUIREMENTS
Students will spend the first eight months (two semesters) of the program completing their technical
graduate courses as well as CSC2701H (Communication for Computer Scientists). CSC2702H
(Technical Entrepreneurship) will be completed during the second eight months, normally in
conjunction with the full-time applied research internship.
All students must complete a minimum of four technical graduate courses, in accordance with their
concentration requirements. These must be equivalent to at least 2.0 Full Course Equivalents (FCEs)
and all students must show satisfactory academic progress (defined as a minimum passing grade of B-
(70%) in each course). If a student has not made satisfactory academic progress by the end of
the second semester, they must immediately contact the Associate Director, MScAC
Administration to determine the options and next steps.
It is an individual student’s responsibility to ensure course selection meets the requirements of the
concentration they are registered in. Degree Explorer should be utilized to ensure requirements are
met. Students who have not met their concentration requirements by the end of the second semester
will be required to take additional courses.
Only students who make satisfactory academic progress in the first two semesters may proceed to the
internship component of the program.
""",

"""
COURSE SELECTION
Applied Mathematics Concentration
Students are required to complete:
• 2 graduate courses (1.0 FCE) from the Department of Computer Science course schedule in
two different groups
• 2 graduate courses (1.0 FCE) from the Department of Mathematics at 1000-level or higher
7
Course from other departments may be eligible to fulfill the mathematics requirements. Course
selection must be approved by the Applied Mathematics Concentration Lead.
Artificial Intelligence Concentration
Students are required to complete:
• Two graduate courses (1.0 FCE) from the core list of Artificial Intelligence courses.
• One graduate course (0.5 FCE) selected from Group 2 (AI) courses outside the core list.
• One graduate course (0.5 FCE) chosen from Groups 1, 3, or 4 (outside of AI).
Students may request a waiver of one AI core course requirement by demonstrating mastery of
equivalent material (usually evidenced through completion of courses at senior undergraduate level).
Waivers are usually applied for during the study plan submission process. All waivers are subject to
approval of the Academic Director, Professional Programs. The waiver does not reduce the number of
courses students are required to take. Instead, it allows students to take an additional AI course outside
the core list. In all cases, students must complete 1.5 FCE in AI courses.
Core Artificial Intelligence Courses
Course Code Course Title
AER1513H State Estimation for Aerospace Vehicles
AER1517H Control for Robotics
CSC2501H Computational Linguistics
CSC2502H Knowledge Representation and Reasoning
CSC2503H Foundations of Computer Vision
CSC2511H Natural Language Computing
CSC2515H* Introduction to Machine Learning (exclusion: ECE1513H)
CSC2516H** Neural Networks and Deep Learning (exclusion: MIE1517H)
CSC2529H Computational Imaging
CSC2533H Foundations of Knowledge Representation
CSC2630H Introduction to Mobile Robotics
ECE1512H Digital Image Processing and Applications
ECE1513H* Introduction to Machine Learning (exclusion: CSC2515H)
MAT1510H Deep Learning, Theory and Data Science
MIE1517H** Introduction to Deep Learning (exclusion: CSC2516H)
Courses from other departments may be eligible to fulfill the non-core AI/non-AI course requirements.
Course selection must be approved by the Associate Director, Academic.
""",

"""
Artificial Intelligence in Healthcare Concentration
Students are required to complete:
• 1 graduate course (0.5 FCE) in Data Science from the approved list.
• 1 graduate course (0.5 FCE) from the approved list of Artificial Intelligence courses.
• 1 graduate course (0.5 FCE) from the approved list of courses in Group 3.
• 1 graduate course (0.5 FCE) from the approved list of LMP/MHI coursework from the approved
list.
Due to the variance of course schedule availability, approved courses will be provided during the study
plan solicitation process. Course selection must be approved by the Artificial Intelligence in Healthcare
Concentration Lead.
8
Computer Science Concentration
Students are required to complete:
• 4 graduate courses (2.0 FCE) from at least 2 of the 4 course groups. A maximum of 1 course
(0.5 FCE) from Group 2 (AI) will be counted towards the program requirements.
• 2 courses (1.0 FCE) must be from the Computer Science timetable (i.e. CSCXXXX course code).
• 2 courses (1.0 FCE) may be chosen from other departments pending approval by the MScAC
program.
Data Science Concentration
Students are required to complete:
• 2 graduate courses (1.0 FCE) from the Department of Computer Science course listings in two
different groups.
• 2 graduate courses (1.0 FCE) chosen from the STA2000-level courses or higher. This must
include STA2453H Data Science Methods, Collaboration and Communication. A maximum of
0.5 FCE may be chosen from the STA4500-level of six-week modular courses. Note that some
courses at STA4500 level and higher are six-week modular courses weighted at 0.25 FCE
each.
Courses from other departments may be eligible to fulfill the statistics requirements. Course selection
must be approved by the Data Science Concentration Lead.
Data Science for Biology
Students are required to complete:
• 2 graduate courses (1.0 FCE) from the Department of Computer Science course listings in two
different groups.
• 1 graduate course (0.5 FCE) from the Department of Cell & Systems Biology.
• 1 graduate course (0.5 FCE) with a focus on computational biology or bioinformatics chosen
from the approved course lists from the following departments: Cell & Systems Biology,
Ecology & Evolutionary Biology, Molecular Genetics or Statistics.
Quantum Computing Concentration
Students are required to complete:
• 2 graduate courses (1.0 FCE) from the Department of Computer Science course listings in two
different groups.
• 2 graduate courses (1.0 FCE) from the Department of Physics.
Courses from other departments may be eligible to fulfil the physics requirements.
""",

"""
Changing Your Concentration
The concentration you have been admitted to is the one the admissions committee felt your academic
background and additional experience were best suited to. Requests to switch concentration will be
handled on a case-by-case basis and are subject to approval by the MScAC Academic Director,
Professional Programs and respective concentration lead.
To request a change of concentration, you should submit your request and supporting rationale in writing
to the Associate Director, MScAC Administration via: https://forms.office.com/r/PzbAjN88aX
""",

"""
THE INTERNSHIP PROCESS
The last eight months of the MScAC program are spent undertaking an applied research internship. This
internship is a formal requirement of the program. The internship can only be started after satisfactory
academic progress in the coursework is confirmed.
Work Permit
All international students must ensure that they hold a valid work permit allowing for full-time employment
with an industry partner in Canada during the internship period (normally May 1 – December 31). Most
students will apply for a co-op work permit. This should be applied for at the same time as the study
permit, so that the required documentation is in place from the start of the program. If the work permit is
not issued on arrival at the port-of-entry (normally Toronto), contact the Associate Director, MScAC
Administration for an updated letter so a work permit application can be submitted in Canada as quickly
as possible.
Details on how to apply for work permits are available from the Immigration and Citizenship Canada
website: canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/intern.html
Safety Abroad
All students who travel outside of Canada for their internship should ensure that they register their travel
in advance with the University of Toronto Safety Abroad Registry. The Safety Abroad office is dedicated
to supporting safer student experiences abroad, and can offer support and emergency assistance.
What is an applied research internship?
An applied research internship usually involves research aggregation, namely the exploration and
synthesis of research results into an evaluation, study, or demonstrable, industrially relevant prototype.
In the service of a company, it is expected that you will leverage your graduate academic training and
past experience to explore new initiatives, improvements in process or product, or new designs that
could be of potential impact.
Your internship may require you to work on explorations that an industry partner company might not
otherwise perform. This requires a higher standard of creative or intellectual exploration than would
normally be encountered in a co-operative (co-op) work term. For example, a role consisting only of
programming tasks would likely not qualify as a research internship. That said, the scope of the MScAC
internship may also involve coding or systems development that leads to a contribution to the company’s
product or service offering.
"""
]

def ask_llm(
    user_prompt: str,
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.5,
    ):
    """
        Generate a response from the Gemini model.

        Args:
            user_prompt: the prompt to generate a response from the Gemini model.
            temperature: the temperature setting passed to the Gemini API

        Returns:
            The response from the Gemini model.
    """

    # Initialize the client
    client = genai.Client(
        api_key=os.environ.get("ADI_GOOGLE_AI_STUDIO_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
    )

    response = ""

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response += chunk.text

    return response


if __name__ == "__main__":
    ask_llm(user_prompt="What is the capital of France?")