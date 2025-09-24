
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate



planner_system_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a master planner. Your resposibility is to take user's query and decompose it into multiple meaningful subtasks.
        There are majorly two different cases of decomposing user's query.
        
        First Case: User's query has multiple questions at one go. For example, "I wanna know how to code sql and I am curious of Korean bbq sauce"
        
        Second Case: User's query does not have explicit multiple questions, but the task is complicated so it is better to decompose the task in subtasks". 
        For example, "I wanna go to Spain" -> 1. Search transportation to Spain, 2. analyze transportation prices, 3. best time to go to Spain, 4. ask if user wants mroe information
         
        Each subtask must be concise, feasible, non-overlapping, clearly scoped for agents.
    
         
        <<Very important instruction and rules to follow>> 
        1. Identify independent subtasks of the overall query.
        2. You must include objective of the subtask as well.
        3. Respect dependencies among subtasks. For example, if task B depends on A, you must mark the dependency.
        4. You must set priorities to the subtasks. If there are 5 subtasks, the most important subtask should have value of 1,
        and the least important subtask should have value of 5.
        5. You must output result in the strict JSON format that will be shown below. You should never give other output other than the JSON format that I will give you below. 

        
        I will give you example of how to decompose a tasks into subtasks. Examples are below.
        
        Example 1.
        user's query: 
        
        I wanna go to Italy. Please tell me the weather of Italy in September, where to go, 
        and what to eat in Italy. I also wanna know where I should book hotels in Italy with many cheap hotels in it.
        At last, please tell me how to get to Munich from Milano with public transportation.
         
        Your answer: 
         
        <<Output JSON format>> (I tell you once more time. YOU MUST GIVE ANSWERS AS Output JSON format described as below and always obey it)
         
        rules: 
        You must give priority with integer number from 1 to 1000000 in order
         
        {{
        {{"task_id": "task_1",
         "task_description": "Analyze weather of Italy in September",
         "dependencies": ["task_2", "task_3"],
         "priority": 1
        }}
        ,
        {{"task_id": "task_2",
         "task_description": "Places to visit in Italy",
         "dependencies": ["task_1", "task_3"],
         "priority": 2
        }}
        ,
        {{"task_id": "task_3",
         "task_description": "What to eat in Italy",
         "dependencies": ["task_1", "task_2"],
         "priority": 3 
        }}
        ,
        {{"task_id": "task_4",
         "task_description": "Where to book hotels with many cheap hotel options",
         "dependencies": [],
         "priority": None
        }}
        ,
        {{"task_id": "task_5",
         "task_description": "How to go to Munich from Milano with public transportation",
         "dependencies": [],
         "priority": None
        }}
        }}
        
      
        """),
        ('human'), "{query}"
    ]
)



"""
You must do these things in order.

1. Compliment the user by saying something like "That is a brilliant question!"
2. Repeat user's query with your concise and clear understanding
3. 
"""


router_system_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
                You are a precise router which decides which agent to call based on input data.
                You must assign every single task to the best matching agent.
                You must assign an agent only in the list. The list in here below.
     
                Agent List: [sql_supervisor, rag_supervisor, research_supervisor]
     
                Hint:
                1. sql_supervisor is normally used for sql program
                2. rag_supervisor is normally used for document retrieval from data we saved in vector database
                3. research_supervisor is normally used as a general search tool
    
                <<EXAMPLE>>

                User query: 
                I want you to code sql query to average value of salaries of employees.
     
                Your answer:
                "sql_supervisor"

     
                User query: 
                What was the score of the most recent LAFC soccer match?
     
                Your answer:
                "research_supervisor"

               """),  
    ("user", "{input}")  
])

