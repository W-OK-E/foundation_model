import numpy as np

def generate_series(start_num, end_num):
    series = []
    current_value = start_num
    increment = 2
    upper_limit = 10

    while current_value <= end_num:
        # If current_value is below the upper_limit, add it to the series
        while current_value <= upper_limit and current_value <= end_num:
            series.append(current_value)
            current_value += increment
        
        # Double the increment and increase the upper limit in larger steps
        increment += 2
        upper_limit += 10

    return series

# Example usage with starting number 3 and ending number 1000000000000
# example_series = generate_series(3, 200)
# example_series

def divide_list_into_n_sublists(lst, n):
    """Divide a list into n sublists as evenly as possible."""
    avg = len(lst) / float(n)
    sublists = []
    last = 0.0

    while last < len(lst):
        sublists.append(lst[int(last):int(last + avg)])
        last += avg

    return sublists

def select_n_elements_with_repeats_and_fallback(series, n, mode = None):
    # Split the series into numbers less than 10 and numbers greater than or equal to 10
    less_than_10 = [num for num in series if num < 10]
    greater_equal_10 = [num for num in series if num >= 10]
    
    # Calculate how many elements should come from each group (half from each)
    half_n = n // 2
    #greater_10 = n - half_n
    
    # If there are fewer elements less than 10, repeat those numbers to meet half_n requirement
    selected_less_than_10 = []
    while len(selected_less_than_10) < half_n:
        selected_less_than_10.extend(less_than_10)
    selected_less_than_10 = selected_less_than_10[:half_n]  # Limit to half_n elements
    
    # Remaining elements to be selected from numbers greater than or equal to 10
    remaining_elements_needed = n - len(selected_less_than_10)
    
    if len(greater_equal_10) >= remaining_elements_needed:
        # Divide greater_equal_10 into subgroups if enough elements are available 
        subgroups = divide_list_into_n_sublists(greater_equal_10, remaining_elements_needed)
        # Select one element randomly from each subgroup
        selected_greater_equal_10 = []
        for group in subgroups:
            if group:  # Make sure the group is not empty
                if mode == 'max':
                    selected_greater_equal_10.append(np.max(group))
                else:
                    selected_greater_equal_10.append(int(np.median(group)))
    else:
        # If there are fewer elements in greater_equal_10 than needed, select all of them
        selected_greater_equal_10 = greater_equal_10
        # Fill the remaining spots with random numbers from less_than_10
        while len(selected_greater_equal_10) < remaining_elements_needed:
            #selected_greater_equal_10.append(random.choice(less_than_10))
            selected_greater_equal_10.extend(less_than_10)
        selected_greater_equal_10 = selected_greater_equal_10[:remaining_elements_needed]
    
    # Combine the two selections
    selected_elements = selected_less_than_10 + selected_greater_equal_10
    
    return selected_elements